import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split


def load_fer2013(csv_path):
    """
    Load FER-2013 dataset from CSV file
    """

    data = pd.read_csv(csv_path)

    pixels = data['pixels'].tolist()
    emotions = data['emotion'].tolist()

    images = []

    for pixel_sequence in pixels:
        image = np.array(pixel_sequence.split(), dtype='uint8')
        image = image.reshape(48, 48)

        # Resize image (optional)
        image = cv2.resize(image, (64, 64))

        images.append(image)

    images = np.array(images)
    emotions = np.array(emotions)

    return images, emotions


def split_data(images, labels, test_size=0.2):
    """
    Split dataset into train and test sets
    """

    X_train, X_test, y_train, y_test = train_test_split(
        images,
        labels,
        test_size=test_size,
        random_state=42,
        stratify=labels
    )

    return X_train, X_test, y_train, y_test