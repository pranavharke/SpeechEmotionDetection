
# Speech Emotion Detection using REVDESS

## Overview
This project implements a Speech Emotion Detection system using deep learning and CNNs. The system is trained on the RAVDESS dataset and can predict emotions from speech audio files. It includes model training, evaluation, and a web-based deployment using Streamlit.

## Features
- **Train a CNN model** to classify emotions from speech.
- **Support for VGG16-based model** for improved accuracy.
- **Streamlit Web App** for real-time emotion detection from user-uploaded audio.
- **Preprocessed dataset** stored in `.npy` format for fast training.

## Installation
Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
### Run the Streamlit App
To launch the web interface:
```bash
streamlit run app/app.py
```

## Dataset Overview
The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) consists of **7,356 files** (Total Size: **24.8 GB**). It includes **24 professional actors** (12 Male, 12 Female) with a neutral North American accent.

### Emotions Included:
- **Neutral** 😐
- **Calm** 😌
- **Happy** 😊
- **Sad** 😞
- **Angry** 😠
- **Fearful** 😨
- **Surprised** 😯
- **Disgust** 🤢

Each emotion is recorded at normal and strong intensities, with an additional neutral expression.

### Formats:
- **Audio:** `.wav` format, 16-bit, 48kHz (High-quality, uncompressed sound)
- **Video:** `.mp4` format (No audio, compressed for efficient storage and streaming)

### File Naming Convention
Each file in RAVDESS follows a structured naming convention:
`02-01-06-01-02-01-12.mp4`
- **Modality:** (01 = Full-AV, 02 = Video-only, 03 = Audio-only)
- **Vocal Channel:** (01 = Speech, 02 = Song)
- **Emotion:** (01 = Neutral, 02 = Calm, 03 = Happy, etc.)
- **Emotional Intensity:** (01 = Normal, 02 = Strong)
- **Statement:** (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door")
- **Repetition:** (01 = 1st repetition, 02 = 2nd repetition)
- **Actor:** (01 to 24. Odd = Male, Even = Female)

### Example
A file named `02-01-06-01-02-01-12.mp4` represents:
- **Modality:** Video-only (02)
- **Speech Type:** Speech (01)
- **Emotion:** Fearful (06)
- **Intensity:** Normal (01)
- **Statement:** "Dogs are sitting by the door" (02)
- **Repetition:** First (01)
- **Actor:** 12 (Female, since even-numbered)

## Project Objective
- It's not what you say, but the way you say it that matters.
- The objective of this project is to develop a machine learning model that can recognize emotions from audio files in the RAVDESS dataset.

### Key Goals:
- **Data Preparation:** Organize and preprocess speech audio.
- **Feature Extraction:** Extract MFCCs and other relevant features.
- **Model Training:** Train deep learning models for emotion classification.
- **Evaluation:** Assess model performance and enhance real-world accuracy.

## Applications
Emotion recognition from audio data has impactful applications across various sectors:
- **Security:** Detects suspicious behaviors in public or high-security areas.
- **Entertainment:** Adapts content based on viewer emotions for a personalized experience.
- **Human-Computer Interaction:** Personalizes virtual assistant interactions based on user mood.
- **Education:** Identifies student engagement and frustration in e-learning for adaptive learning.
- **Healthcare:** Monitors emotional well-being of non-verbal or cognitively impaired patients.
- **Mental Health:** Tracks emotional states in virtual therapy to detect stress, anxiety, or depression.
- **Customer Service:** Enhances call center interactions by detecting emotions in real-time.

## Future Improvements
- Implement real-time emotion detection using microphone input.



