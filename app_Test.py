


import streamlit as st
import numpy as np
import joblib
import librosa
import soundfile as sf
import sounddevice as sd
import tempfile
import wavio
from utils import extract_mfcc

# ---------------------------------------------------
# 🔹 Load model and labels
# ---------------------------------------------------
model = joblib.load("model.pkl")

emotion_labels = {
    0: "angry",
    1: "calm",
    2: "happy",
    3: "sad",
    4: "neutral",
    5: "fearful",
    6: "disgust",
    7: "surprised"
}

# ---------------------------------------------------
# 🔹 Streamlit UI setup
# ---------------------------------------------------
st.set_page_config(page_title="Speech Emotion Detector", layout="centered")
st.title("🎤 Speech Emotion Recognition App")

st.write("Upload a `.wav` file or record live speech to detect the emotion 👇")


# ---------------------------------------------------
# 🔹 Upload Audio File
# ---------------------------------------------------
st.subheader("📁 Upload an Audio File")

uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    try:
        audio_data, sample_rate = sf.read(uploaded_file)
        mfcc = extract_mfcc(audio_data, sample_rate)
        prediction = model.predict(mfcc)[0]
        emotion = emotion_labels.get(int(prediction), "Unknown")

        if emotion == "Unknown":
            st.warning(f"⚠️ Model predicted label {int(prediction)}, which isn't in your mapping.")
        else:
            st.success(f"🧠 Detected Emotion: **{emotion.upper()}**")

    except Exception as e:
        st.error(f"⚠️ Error processing the audio file: {str(e)}")


# ---------------------------------------------------
# 🔹 Live Recording
# ---------------------------------------------------
st.subheader("🎙️ Record Live Speech")

duration = st.slider("Recording duration (seconds)", 3, 10, 4)
fs = 22050  

if st.button("🔴 Start Recording"):
    try:
        st.info("Recording... Speak now 🎧")
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
        sd.wait()
        st.success("✅ Recording complete!")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
            wavio.write(tmpfile.name, recording, fs, sampwidth=2)
            audio_path = tmpfile.name

        st.audio(audio_path)

        signal, sr = librosa.load(audio_path, sr=fs)
        mfcc = extract_mfcc(signal, sr)

        prediction = model.predict(mfcc)[0]
        emotion = emotion_labels.get(int(prediction), "Unknown")

        if emotion == "Unknown":
            st.warning(f"⚠️ Model predicted label {int(prediction)}, which isn't in your mapping.")
        else:
            st.success(f"🎯 Predicted Emotion from Live Speech: **{emotion.upper()}**")

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {str(e)}")


# ---------------------------------------------------
# ✅ Footer
# ---------------------------------------------------
st.markdown("---")
st.markdown("💡 If 'Unknown' appears, check model.classes_ and update label mapping.")
