from Mask_RCNN.mrcnn.config import Config


class FoodConfig(Config):
    # Give the configuration a recognizable name
    NAME = "food"
    # Number of classes (including background)
    NUM_CLASSES = 1 + 6  # Background + onion
    # NUMBER OF GPUs to use. When using only a CPU, this needs to be set to 1.
    GPU_COUNT = 1
    # Backbone network architecture
    # Supported values are: resnet50, resnet101.
    # You can also provide a callable that should have the signature
    # of model.resnet_graph. If you do so, you need to supply a callable
    # to COMPUTE_BACKBONE_SHAPE as well
    BACKBONE = "resnet50"

    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024

    # Number of images to train with on each GPU. A 12GB GPU can typically
    # handle 2 images of 1024x1024px.
    # Adjust based on your GPU memory and image sizes. Use the highest
    # number that your GPU can handle for best performance.
    IMAGES_PER_GPU = 2
