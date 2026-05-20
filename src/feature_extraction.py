import numpy as np
from skimage.feature import hog


def extract_hog_features(images):
    """
    Extract HOG features from a list/array of grayscale images.
    """

    hog_features = []

    for image in images:
        features = hog(
            image,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            visualize=False
        )

        hog_features.append(features)

    return np.array(hog_features)