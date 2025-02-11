import os
import io
import wave
import threading
import numpy as np
import librosa
import tensorflow as tf
import streamlit as st
import pyaudio
import cv2

# Define model directory and file path
MODEL_DIR = "models"
MODEL_FILE = "speech_emotion_vgg16_model.h5"  # Ensure this matches the filename in your folder
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)

# Ensure model directory exists
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Check if model file exists
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at {MODEL_PATH}. Please upload the correct model below.")
    st.stop()  # Stop execution if model is missing

# Load the trained VGG16-based emotion detection model
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Audio recording parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024

audio = pyaudio.PyAudio()
recording = False
frames = []

# Function to extract Mel Spectrogram as a 2D image for VGG16
def extract_mel_spectrogram(audio_bytes, img_size=(224, 224)):
    try:
        audio_stream = io.BytesIO(audio_bytes)
        with wave.open(audio_stream, 'rb') as wf:
            sr = wf.getframerate()
            frames = wf.readframes(wf.getnframes())

        y = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Resize to VGG16 input size (224x224)
        mel_spec_resized = cv2.resize(mel_spec_db, img_size, interpolation=cv2.INTER_LINEAR)

        # Normalize to range [0,1]
        mel_spec_resized = (mel_spec_resized - mel_spec_resized.min()) / (mel_spec_resized.max() - mel_spec_resized.min())

        # Convert to 3-channel image for VGG16 (RGB-like format)
        mel_spec_rgb = np.stack([mel_spec_resized] * 3, axis=-1)

        return np.expand_dims(mel_spec_rgb, axis=0)  # Add batch dimension
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None

# Function to start recording
def start_recording():
    global recording, frames
    recording = True
    frames = []

    def record():
        stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                            input=True, frames_per_buffer=CHUNK)
        while recording:
            data = stream.read(CHUNK)
            frames.append(data)
        stream.stop_stream()
        stream.close()

    thread = threading.Thread(target=record)
    thread.start()

# Function to stop recording and process audio
def stop_recording():
    global recording
    recording = False

    # Convert recorded frames to a bytes object
    audio_bytes = b''.join(frames)

    # Create an in-memory WAV file
    audio_stream = io.BytesIO()
    with wave.open(audio_stream, 'wb') as wave_file:
        wave_file.setnchannels(CHANNELS)
        wave_file.setsampwidth(audio.get_sample_size(FORMAT))
        wave_file.setframerate(RATE)
        wave_file.writeframes(audio_bytes)

    return audio_stream.getvalue()  # Return audio bytes

# Streamlit UI
st.title("🎙️ Speech Emotion Detection")

st.write("🎤 Click **Start Recording** to begin speaking, and **Stop Recording** to analyze.")

# Create Streamlit buttons
if st.button("Start Recording 🎙️"):
    start_recording()
    st.write("Recording... Speak now!")

if st.button("Stop Recording ⏹️"):
    audio_bytes = stop_recording()

    # Extract features and make prediction
    features = extract_mel_spectrogram(audio_bytes)
    if features is not None:
        prediction = model.predict(features)
        emotion_label = np.argmax(prediction)

        # Emotion mapping
        emotion_map = {
            0: "neutral", 1: "calm", 2: "happy", 3: "sad",
            4: "angry", 5: "fearful", 6: "disgust", 7: "surprised"
        }
        st.write(f"Predicted Emotion: **{emotion_map.get(emotion_label, 'Unknown')}**")

# File Upload Option
st.write("---")
st.write("📂 **Upload an audio file** for emotion detection.")

uploaded_file = st.file_uploader("Upload an audio file (.wav)", type=["wav"])

if uploaded_file:
    audio_bytes = uploaded_file.read()

    features = extract_mel_spectrogram(audio_bytes)
    if features is not None:
        prediction = model.predict(features)
        emotion_label = np.argmax(prediction)

        emotion_map = {
            0: "neutral", 1: "calm", 2: "happy", 3: "sad",
            4: "angry", 5: "fearful", 6: "disgust", 7: "surprised"
        }
        st.write(f"Predicted Emotion: **{emotion_map.get(emotion_label, 'Unknown')}**")

# Upload model option if missing
st.write("---")
st.write("📤 **If your model file is missing, upload it here.**")

uploaded_model = st.file_uploader("Upload your trained model (.h5)", type=["h5"])
if uploaded_model:
    model_path = os.path.join(MODEL_DIR, uploaded_model.name)
    with open(model_path, "wb") as f:
        f.write(uploaded_model.getbuffer())
    st.success(f"Model uploaded successfully: {uploaded_model.name}")
    st.write("🔄 **Restart the app to load the new model.**")

st.write("\n" * 10)
st.write("---")
st.write("**Project by**")
st.write("[🔗](https://www.linkedin.com/in/pranavharke) [Pranav Harke](https://github.com/pranavharke)")
st.write("[🔗](https://www.linkedin.com/in/parvejkhan-pathan-891527337/)  Parvej Pathan")
st