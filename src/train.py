import os
import tensorflow as tf
from mrcnn import model as modellib
from config import GrassConfig
from dataset import GrassDataset

def train():
    config = GrassConfig()
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=config.MODEL_DIR)
    model.load_weights(config.COCO_MODEL_PATH, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

    train_set = GrassDataset()
    train_set.load_dataset(os.path.join(config.ROOT_DIR, "data"), is_train=True)
    train_set.prepare()

    test_set = GrassDataset()
    test_set.load_dataset(os.path.join(config.ROOT_DIR, "data"), is_train=False)
    test_set.prepare()

    model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=100, layers='heads')

if __name__ == "__main__":
    train()