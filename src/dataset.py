import os
import xml.etree.ElementTree as ET
import numpy as np
import skimage.io
from imgaug.augmentables.polys import Polygon, PolygonsOnImage
from mrcnn import utils

class GrassDataset(utils.Dataset):
    def __init__(self, class_map=None):
        super().__init__(class_map)
        self.augmentations = []

    def read_xml(self, filename):
        tree = ET.parse(filename)
        root = tree.getroot()
        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
        depth = int(root.find('.//size/depth').text)
        polygons = []
        names = []
        for obj in root.findall('object'):
            names.append(obj.find('name').text)
            for polygon in obj.iter(tag='polygon'):
                x_coord = []
                y_coord = []
                for coord in polygon.getchildren():
                    x_coord.append(int(coord.text)) if coord.tag == 'x' else y_coord.append(int(coord.text))
                polygons.append(Polygon(list(zip(x_coord, y_coord)), names[-1]))
        return width, height, depth, polygons

    def load_dataset(self, dataset_dir, is_train=True):
        self.add_class("dataset", 1, "grass")
        images_dir = os.path.join(dataset_dir, 'images')
        annotations_dir = os.path.join(dataset_dir, 'annotations')
        for filename in os.listdir(images_dir):
            if filename == '.DS_Store':
                continue
            image_id = filename[:-4]
            if is_train and int(image_id) >= 52:
                continue
            if not is_train and int(image_id) < 52:
                continue
            img_path = os.path.join(images_dir, filename)
            ann_path = os.path.join(annotations_dir, f"{image_id}.xml")
            w, h, d, polygons = self.read_xml(ann_path)
            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path, height=h, width=w, depth=d, polygons=polygons)

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        w, h = info['width'], info['height']
        polygons = info['polygons']
        mask = np.zeros([h, w, len(polygons)], dtype='uint8')
        class_ids = []
        for i, polygon in enumerate(polygons):
            rr, cc = skimage.draw.polygon(polygon.yy_int, polygon.xx_int)
            rr[rr > mask.shape[0]-1] = mask.shape[0]-1
            cc[cc > mask.shape[1]-1] = mask.shape[1]-1
            mask[rr, cc, i] = 1
            class_ids.append(self.class_names.index(polygon.label))
        return mask, np.array(class_ids, dtype='int32')