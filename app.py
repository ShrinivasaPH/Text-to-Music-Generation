import streamlit as st
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import soundfile as sf
import tempfile
import os

# Set Streamlit page config
st.set_page_config(page_title="Text-to-Music BGM Generator", layout="centered")

st.title("🎶 Text-to-Music BGM Generator")
st.markdown("Generate background scores for narration using Meta’s MusicGen.")

# Sidebar options
st.sidebar.header("Settings")
model_choice = st.sidebar.selectbox(
    "Choose MusicGen Model",
    ["facebook/musicgen-small", "facebook/musicgen-medium", "facebook/musicgen-large"]
)

length = st.sidebar.slider("Music length (seconds)", min_value=5, max_value=30, value=15)

# Load model + processor (cached)
@st.cache_resource
def load_model(model_name):
    model = MusicgenForConditionalGeneration.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor

model, processor = load_model(model_choice)

# Text input
prompt = st.text_area("🎤 Enter your prompt:", 
    "Calm and spiritual background music with bansuri flute and soft tabla, suitable for Ramayana narration."
)

if st.button("Generate Music"):
    if prompt.strip() == "":
        st.warning("Please enter a description.")
    else:
        with st.spinner("🎼 Generating music..."):
            inputs = processor(
                text=[prompt],
                padding=True,
                return_tensors="pt",
            )

            with torch.no_grad():
                audio_values = model.generate(**inputs, max_new_tokens=length * 20)  
                # approx 20 tokens per sec

            # Save temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                sf.write(tmp.name, audio_values[0, 0].cpu().numpy(), model.config.audio_encoder.sampling_rate)
                audio_path = tmp.name

        st.success("✅ Music generated!")
        st.audio(audio_path, format="audio/wav")

        with open(audio_path, "rb") as f:
            st.download_button("⬇️ Download Music", f, file_name="generated_bgm.wav", mime="audio/wav")
