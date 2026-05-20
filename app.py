import streamlit as st
import cv2
import joblib
import numpy as np
from PIL import Image
from skimage.feature import hog


EMOTION_LABELS = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}


@st.cache_resource
def load_model():
    return joblib.load("models/svm_emotion_model.pkl")


def extract_hog(image):
    features = hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=False
    )
    return features.reshape(1, -1)


def predict_face(gray_face, model):
    resized = cv2.resize(gray_face, (64, 64))
    features = extract_hog(resized)

    prediction = model.predict(features)[0]
    emotion = EMOTION_LABELS[prediction]

    confidence_scores = {}

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        confidence_scores = {
            EMOTION_LABELS[i]: float(probabilities[i]) * 100
            for i in range(len(probabilities))
        }

    return emotion, confidence_scores


def detect_faces(gray):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    return faces


def predict_uploaded_image(uploaded_image, model):
    image = Image.open(uploaded_image).convert("RGB")
    image_np = np.array(image)

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    faces = detect_faces(gray)

    if len(faces) == 0:
        emotion, confidence_scores = predict_face(gray, model)
        return image_np, emotion, confidence_scores

    emotion = "Unknown"
    confidence_scores = {}

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        emotion, confidence_scores = predict_face(face, model)

        cv2.rectangle(image_np, (x, y), (x+w, y+h), (34, 197, 94), 3)
        cv2.putText(
            image_np,
            str(emotion),
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (34, 197, 94),
            2
        )

    return image_np, emotion, confidence_scores

def webcam_prediction(model):
    st.subheader("Live Webcam Emotion Detection")

    start = st.checkbox("Start Webcam", key="webcam_start_checkbox")

    left_col, right_col = st.columns([1.2, 1])

    with left_col:
        frame_placeholder = st.empty()

    with right_col:
        prediction_placeholder = st.empty()
        confidence_placeholder = st.empty()

    if start:
        camera = cv2.VideoCapture(0)

        if not camera.isOpened():
            st.error("Could not open webcam.")
            return

        while start:
            ret, frame = camera.read()

            if not ret:
                st.error("Failed to capture frame.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detect_faces(gray)

            current_emotion = None
            current_confidence_scores = {}

            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                emotion, confidence_scores = predict_face(face, model)

                current_emotion = emotion
                current_confidence_scores = confidence_scores

                cv2.rectangle(frame, (x, y), (x+w, y+h), (34, 197, 94), 3)
                cv2.putText(
                    frame,
                    str(emotion),
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (34, 197, 94),
                    2
                )

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame_placeholder.image(
                frame,
                channels="RGB",
                width=520
            )

            if current_emotion is not None:
                prediction_placeholder.markdown(
                    f"""
                    <div class="prediction-box">
                        Live Prediction: {current_emotion}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                with confidence_placeholder.container():
                    st.markdown("### Confidence Scores")

                    for label, score in current_confidence_scores.items():
                        st.write(f"{label}: {score:.2f}%")
                        st.progress(min(score / 100, 1.0))
            else:
                prediction_placeholder.info("No face detected.")

        camera.release()


def main():
    st.set_page_config(
        page_title="Facial Emotion Detection",
        page_icon="😊",
        layout="wide"
    )

    st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
    }

    .hero-card {
        padding: 35px;
        border-radius: 25px;
        background: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        margin-bottom: 30px;
    }

    .title {
        font-size: 48px;
        font-weight: 800;
        color: #1f2937;
        line-height: 1.1;
    }

    .subtitle {
        font-size: 18px;
        color: #6b7280;
        margin-top: 15px;
    }

    .feature-box {
        padding: 22px;
        border-radius: 18px;
        background: #ffffff;
        box-shadow: 0 8px 20px rgba(0,0,0,0.06);
        text-align: center;
        min-height: 150px;
    }

    .prediction-box {
        padding: 25px;
        border-radius: 20px;
        background: #ecfdf5;
        border: 1px solid #10b981;
        font-size: 24px;
        font-weight: 700;
        color: #065f46;
        text-align: center;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

    model = load_model()

    with st.sidebar:
        st.markdown("## FED App")
        st.write("Choose how you want to test the model.")

        option = st.radio(
            "Prediction Mode",
            ["Upload Image", "Live Webcam"]
        )

        st.markdown("---")
        st.markdown("### Tech Stack")
        st.write("Python")
        st.write("OpenCV")
        st.write("Scikit-learn")
        st.write("HOG Features")
        st.write("Streamlit")

    st.markdown("""
    <div class="hero-card">
        <div class="title">Facial Emotion Detection</div>
        <div class="subtitle">
            Detect emotions from facial images using SVM + HOG features.
            Upload an image or use your webcam for real-time prediction.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="feature-box">
            <h3>Image Upload</h3>
            <p>Predict emotion from a face image.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-box">
            <h3>Live Webcam</h3>
            <p>Detect faces and emotions live.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-box">
            <h3>SVM Model</h3>
            <p>Classical ML with HOG features.</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    if option == "Upload Image":
        st.subheader("Upload Image Prediction")

        uploaded_file = st.file_uploader(
            "Choose a face image",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            if st.button("Predict Emotion"):
                result_image, emotion, confidence_scores = predict_uploaded_image(
                    uploaded_file,
                    model
                )

                left_col, right_col = st.columns([1.2, 1])

                with left_col:
                    st.image(
                        result_image,
                        caption="Processed Image",
                        width=520
                    )

                with right_col:
                    st.markdown(
                        f"""
                        <div class="prediction-box">
                            Predicted Emotion: {emotion}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    if confidence_scores:
                        st.markdown("### Confidence Scores")

                        for label, score in confidence_scores.items():
                            st.write(f"{label}: {score:.2f}%")
                            st.progress(min(score / 100, 1.0))
                    else:
                        st.info(
                            "Confidence scores require retraining the model with SVC(probability=True)."
                        )

    elif option == "Live Webcam":
        webcam_prediction(model)

if __name__ == "__main__":
    main()