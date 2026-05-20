import os
import joblib

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from preprocessing import load_fer2013, split_data
from feature_extraction import extract_hog_features


EMOTION_LABELS = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral"
}


def train_model(csv_path):
    print("Loading FER-2013 dataset...")
    images, labels = load_fer2013(csv_path)

    # Reduce dataset size for memory efficiency
    images = images[:10000]  # samples per class
    labels = labels[:10000]  # samples per class

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = split_data(images, labels)

    print("Extracting HOG features...")
    X_train_hog = extract_hog_features(X_train)
    X_test_hog = extract_hog_features(X_test)

    print("Training SVM models with GridSearchCV...")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(probability=True))
    ])

    param_grid = [
        {
            "svm__kernel": ["linear"],
            "svm__C": [1]
        },
        {
            "svm__kernel": ["rbf"],
            "svm__C": [1],
            "svm__gamma": ["scale"]
        }
    ]

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring="f1_weighted",
        verbose=2,
        n_jobs=1
    )

    grid_search.fit(X_train_hog, y_train)

    best_model = grid_search.best_estimator_

    print("\nBest Parameters:")
    print(grid_search.best_params_)

    print("\nEvaluating model...")
    y_pred = best_model.predict(X_test_hog)

    accuracy = accuracy_score(y_test, y_pred)

    print("\nAccuracy:", accuracy)

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=[EMOTION_LABELS[i] for i in range(7)]
    ))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    os.makedirs("models", exist_ok=True)

    model_path = "models/svm_emotion_model.pkl"
    joblib.dump(best_model, model_path)

    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    train_model("data/fer2013.csv")