import os

class GrassConfig:
    NAME = "grass_config"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  # Background + grass
    STEPS_PER_EPOCH = 131
    LEARNING_RATE = 0.0009
    DETECTION_MIN_CONFIDENCE = 0.9
    MAX_GT_INSTANCES = 10

    def __init__(self):
        self.ROOT_DIR = os.getcwd()
        self.MODEL_DIR = os.path.join(self.ROOT_DIR, "logs")
        self.COCO_MODEL_PATH = os.path.join(self.ROOT_DIR, "models", "mask_rcnn_coco.h5")
        self.IMAGE_DIR = os.path.join(self.ROOT_DIR, "data", "images")