import argparse
from mrcnn import model as modellib, visualize
from config import GrassConfig
from keras.preprocessing.image import load_img, img_to_array

def inference(image_path):
    config = GrassConfig()
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=config.MODEL_DIR)
    model.load_weights(config.COCO_MODEL_PATH, by_name=True)

    img = load_img(image_path)
    img = img_to_array(img)
    results = model.detect([img], verbose=1)
    r = results[0]
    visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], ["BG", "grass"], r['scores'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True, help='Path to the image file')
    args = parser.parse_args()
    inference(args.image_path)