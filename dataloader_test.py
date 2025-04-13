from PIL import Image
from options import BaseOptions

from dataloader.nir_preprocessing import save_nir_gaussian_values
from dataloader.dataloader import get_dataloader


if __name__ == '__main__':
    opts = BaseOptions()
    opts.batch_size = 1
    dataloader = get_dataloader(opts, "all", False, preprocess=False, augment=False)
    print("Dataloader loaded")

    save_nir_gaussian_values(dataloader)
