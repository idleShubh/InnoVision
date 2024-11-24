import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import numpy as np
import speech_recognition as sr

# Load Go Emotions RoBERTa model and tokenizer
model_name = "SamLowe/roberta-base-go_emotions"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create Streamlit UI
st.title("Emotion Analysis with RoBERTa")
st.write("Enter text or use voice input to analyze emotions")

# Initialize speech recognizer
recognizer = sr.Recognizer()

# Add voice input option
def get_voice_input():
    try:
        import pyaudio
    except ImportError:
        st.error("PyAudio is not installed. Please install it using: pip install pyaudio")
        return None
        
    with sr.Microphone() as source:
        st.write("Listening... Speak now")
        try:
            audio = recognizer.listen(source, timeout=5)
            st.write("Processing speech...")
            text = recognizer.recognize_google(audio)
            return text
        except sr.WaitTimeoutError:
            st.warning("No speech detected. Please try again.")
            return None
        except sr.UnknownValueError:
            st.warning("Could not understand audio. Please try again.")
            return None
        except sr.RequestError:
            st.error("Could not connect to speech recognition service.")
            return None

# Text input area
text_input = st.text_area("Input your text here:", height=100)

# Voice input button
if st.button("🎤 Start Voice Input"):
    voice_text = get_voice_input()
    if voice_text:
        text_input = voice_text
        st.session_state['text_input'] = text_input
        st.rerun()

if st.button("Analyze Emotions"):
    if text_input:
        # Tokenize input
        encoded_text = tokenizer(text_input, return_tensors='pt')
        
        # Get prediction
        output = model(**encoded_text)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        
        # Create emotions dictionary
        emotions = {
            0: "admiration 🤩",
            1: "amusement 😄", 
            2: "anger 😠",
            3: "annoyance 😒",
            4: "approval 👍",
            5: "caring 🤗",
            6: "confusion 😕",
            7: "curiosity 🤔",
            8: "desire 😍",
            9: "disappointment 😞",
            10: "disapproval 👎",
            11: "disgust 🤢",
            12: "embarrassment 😳",
            13: "excitement 🤪",
            14: "fear 😨",
            15: "gratitude 🙏",
            16: "grief 😢",
            17: "joy 😊",
            18: "love ❤️",
            19: "nervousness 😰",
            20: "optimism 🌟",
            21: "pride 🦁",
            22: "realization 💡",
            23: "relief 😌",
            24: "remorse 😔",
            25: "sadness 😭",
            26: "surprise 😲",
            27: "neutral 😐"
        }
        
        # Display results in a nice format
        st.write("\n### Emotion Analysis Results:")
        
        # Show top 5 emotions
        top_emotions = np.argsort(scores)[-5:][::-1]
        
        cols = st.columns(5)
        for idx, (col, emotion_idx) in enumerate(zip(cols, top_emotions)):
            with col:
                emotion = emotions[emotion_idx]
                score = scores[emotion_idx]
                st.metric(emotion, f"{score*100:.1f}%")
            
        # Determine dominant emotion
        dominant_emotion = emotions[np.argmax(scores)]
        st.write(f"\n### Primary Emotion: {dominant_emotion}")
        
        # Check for mixed emotions
        if len([s for s in scores if s > 0.2]) > 2:
            st.info("ℹ️ Multiple strong emotions detected - this text appears to express complex feelings!")
            
    else:
        st.warning("Please enter some text or use voice input to analyze!")

st.markdown("""
---
Note: This emotion analyzer uses the RoBERTa model trained on the GoEmotions dataset.
It can detect 27 different emotions plus neutral, providing detailed insight into the emotional content of text.
You can either type your text or use the voice input feature to analyze emotions.
""")
