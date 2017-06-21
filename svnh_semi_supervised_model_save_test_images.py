import os
import numpy as np
import scipy.misc

import utils

'''
Extracts random 64 test images from the Matlab format and save them
onto the disk
'''

def main():
    images_dir = 'svnh_test_images'
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    utils.download_train_and_test_data()
    _, testset = utils.load_data_sets()

    idx = np.random.randint(0, testset['X'].shape[3], size=64)
    test_images = testset['X'][:, :, :, idx]

    test_images = np.rollaxis(test_images, 3)
    for ii in range(len(test_images)):
        scipy.misc.toimage(test_images[ii]).save("{}/image_{}.jpg".format(images_dir, ii))


if __name__ == '__main__':
    main()
