

class Config(object):

    # The config of network
    DEPTH = 40
    GROWTH_RATE = 12
    COMPRESSION = 1.0
    NUM_DENSE_BLOCK = 3
    BOTTLENECK = False
    NUM_INIT_FILTER = 16
    SUB_SAMPLE_IMAGE = False # if dealing with ImageNet is True
    NUM_CLASSES = 10
    NET_NAME = "DenseNet40"

    # The config of dealing data
    DATA_FORMAT = "channels_first"
    LOG_OUTPUT_DIR = "./logs"
    NUM_ITEM_DATASET = 50000
    IMAGE_MEANS = [125.3, 123.0, 113.9]
    IMAGE_STD = [63.0,  62.1,  66.7]

    # The config of training model
    DROPOUT_RATES = 0.0
    NUM_GPU = 1
    EPOCH = 300
    BATCH_SIZE = 128  # per GPU
    WEIGHT_DECAY = 1e-4
    INIT_LEARNING_RATE = 1e-1
    EPOCH_BOUNDARY = [150, 225]

    def __init__(self):
        self.BOUNDARY = [self.NUM_ITEM_DATASET * i // self.BATCH_SIZE for i in self.EPOCH_BOUNDARY]
        self.SAVE_EVERY_N_STEP = int(self.NUM_ITEM_DATASET / self.BATCH_SIZE)


class Cifar10Config(Config):

    LABEL_BYTES = 1
    IMAGE_SIZE = 32
    CHANNELS = 3

    def __init__(self):
        Config.__init__(self)
