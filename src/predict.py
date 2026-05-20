import cv2
import joblib
import numpy as np
from skimage.feature import hog


EMOTION_LABELS = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral"
}


def preprocess_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Image not found. Check the image path.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))

    return resized


def extract_single_hog(image):
    features = hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=False
    )

    return features.reshape(1, -1)


def predict_emotion(image_path, model_path="models/svm_emotion_model.pkl"):
    model = joblib.load(model_path)

    image = preprocess_image(image_path)
    features = extract_single_hog(image)

    prediction = model.predict(features)[0]

    return EMOTION_LABELS[prediction]


if __name__ == "__main__":
    image_path = input("Enter image path: ")
    emotion = predict_emotion(image_path)

    print("Predicted Emotion:", emotion)