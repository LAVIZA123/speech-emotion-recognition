import streamlit as st
import numpy as np
import joblib
import librosa
import soundfile as sf
from utils import extract_mfcc

# Load model
model = joblib.load("model.pkl")
emotion_labels = {0: "angry", 1: "calm", 2: "happy", 3: "sad"}
# sir mera model number me predict krta h usko m emotion name me cnvrt krti hu

st.set_page_config(page_title="Speech Emotion Detector", layout="centered")
st.title("🎤 Speech Emotion Recognition App")
# Yeh app ka title aur layout set karta hai

st.write("Upload a `.wav` audio file to detect the emotion in the speech.")


uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])
# yeh part app ka interface bnata h


# Ye user ko allow karta hai audio file upload karne ke liye

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')  
    try:
        # Read and process audio
        audio_data, sample_rate = sf.read(uploaded_file)
        # audio file ko read krke numerical form me convert krti h
        # audio_data = actual waveform value 
        #sampel_rate= audio kitne Hz pe recorded hai

        mfcc = extract_mfcc(audio_data, sample_rate)
        # Ye tumhare audio se MFCC features nikalta hai.

        # Predict emotion
        prediction = model.predict(mfcc)[0]
        emotion = emotion_labels[int(prediction)]
        st.success(f"🧠 Detected Emotion: **{emotion}**")
        # “Sir, yeh code model se prediction leta hai, 
        # usko emotion name me convert karta hai, aur Streamlit app par result show karta hai.”

    except Exception as e:
        st.error(f"Error processing the audio file: {str(e)}")

import streamlit as st
import numpy as np
import librosa
import joblib

# Title
st.title("🎤 Speech Emotion Detection App")

# Load model
model = joblib.load("model.pkl")

# Upload audio file
uploaded_file = st.file_uploader("Upload your voice (.wav)", type=["wav"])

# When user uploads file
if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    # Extract features from the uploaded file
    y, sr = librosa.load(uploaded_file, sr=None)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    features = mfccs.reshape(1, -1)

    # Predict emotion
    prediction = model.predict(features)[0]

    # Show result
    st.success(f"Predicted Emotion: **{prediction}** 🎯")



# “Yeh code audio upload karta hai, MFCC features nikalta hai,
#  trained model ko use karke emotion predict karta hai, aur Streamlit pe result show karta hai.”