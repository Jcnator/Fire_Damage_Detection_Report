import abc
import numpy as np

class BaseOptions(metaclass=abc.ABCMeta):
    def __init__(self):
        self.device = 'cuda'
        self.expdir = ''
        self.debug = True
        self.save_dir = "./results"

        #Training Options
        self.save_freq = 25
        self.log_freq = 1
        self.cross_validation = False
        self.k_fold = 5

        # Validation Options
        self.validation_set  = "validation"

        # Model Options
        self.arch = "ResNet18"
        self.num_channels = 4 # rgb + NIR (near infrared)
        self.num_classes = 3

        # HyperParams
        self.lr = 1e-4
        self.num_epochs = 100
        self.classes = np.array([0, 10, 100])
        self.class_count = np.array([256, 46, 794])
        self.dropout = 0.1
        self.class_weights = True
        self.NIR_buckets = 12

        # Data Options
        self.data_dir = './data'
        self.test_train_split = [0.8, 0.1, 0.1]
        self.batch_size = 16
        self.image_size = 227

        # Logging Options
        self.logging = True
        self.project_name = "FireDamage"
        self.save_images = False

        # Augmentations
        self.augment = True
        self.split_rgb_and_nir = False
        self.standard_augment = True
        self.jpeg = True
        self.cutout = True
        self.noise = True
        self.blur = True
        self.gaussian_nir = False

    def add_aditional_opts(self, args):
        for key, value in vars(self).items():
            #print(key, value)
            if not hasattr(args, key):
                setattr(args, key, value)
        return args
                




class AlexNetOptions(BaseOptions):
    def __init__(self):
        super(AlexNetOptions, self).__init__()

