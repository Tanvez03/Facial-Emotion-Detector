# Facial Emotion Detection using SVM

Download FER-2013 from Kaggle and place fer2013.csv inside data/.
Train locally to generate models/svm_emotion_model.pkl.



This project detects facial emotions using classical machine learning.

It uses:

- FER-2013 dataset
- Image preprocessing
- HOG feature extraction
- StandardScaler
- Support Vector Machine
- GridSearchCV
- Streamlit web app

## Emotions Detected

- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

## Project Structure

```text
facial-emotion-detection-svm/
│
├── data/
├── src/
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── train_model.py
│   └── predict.py
├── models/
├── app.py
├── requirements.txt
└── README.md
