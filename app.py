import streamlit as st
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import soundfile as sf
import tempfile
import os
import gc

# Set Streamlit page config
st.set_page_config(page_title="Text-to-Music BGM Generator", layout="centered")

st.title("🎶 Text-to-Music BGM Generator")
st.markdown("Generate background scores for narration using Meta's MusicGen.")

# Check device availability
device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.info(f"Using device: {device}")

# Sidebar options
st.sidebar.header("Settings")
model_choice = st.sidebar.selectbox(
    "Choose MusicGen Model",
    ["facebook/musicgen-small", "facebook/musicgen-medium"],  # Removed large for performance
    help="Small model is fastest, Medium provides better quality"
)

length = st.sidebar.slider("Music length (seconds)", min_value=5, max_value=30, value=15)

# Advanced settings
with st.sidebar.expander("Advanced Settings"):
    use_fp16 = st.checkbox("Use FP16 (faster, less memory)", value=True if device == "cuda" else False)
    batch_size = st.selectbox("Batch size", [1], disabled=True)  # Keep at 1 for now

# Load model + processor (cached and optimized)
@st.cache_resource
def load_model(model_name, device, use_fp16):
    """Load and optimize model for faster inference"""
    try:
        # Load model
        model = MusicgenForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if use_fp16 and device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
        processor = AutoProcessor.from_pretrained(model_name)
        
        # Move to device
        model = model.to(device)
        
        # Optimize for inference
        model.eval()
        if device == "cuda":
            model = torch.compile(model, mode="reduce-overhead")  # PyTorch 2.0+ optimization
            
        return model, processor
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Initialize model
with st.spinner("🔄 Loading model (this may take a moment)..."):
    model, processor = load_model(model_choice, device, use_fp16)

if model is None:
    st.error("Failed to load model. Please try again.")
    st.stop()

st.success(f"✅ Model loaded successfully on {device}")

# Text input
prompt = st.text_area("🎤 Enter your prompt:", 
    value="Calm and spiritual background music with bansuri flute and soft tabla, suitable for Ramayana narration.",
    help="Be specific about instruments, mood, and style for better results"
)

# Generation parameters
col1, col2 = st.columns(2)
with col1:
    guidance_scale = st.slider("Guidance Scale", 1.0, 10.0, 3.0, 0.5, 
                              help="Higher values follow prompt more closely")
with col2:
    temperature = st.slider("Temperature", 0.1, 2.0, 1.0, 0.1,
                           help="Controls randomness - lower is more focused")

if st.button("🎵 Generate Music", type="primary"):
    if prompt.strip() == "":
        st.warning("Please enter a description.")
    else:
        # Calculate tokens more accurately
        tokens_per_second = 50  # More accurate for MusicGen
        max_tokens = int(length * tokens_per_second)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("🔄 Processing prompt...")
            progress_bar.progress(20)
            
            # Process inputs
            inputs = processor(
                text=[prompt],
                padding=True,
                return_tensors="pt",
            )
            
            # Move inputs to device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            status_text.text("🎼 Generating music...")
            progress_bar.progress(40)
            
            # Generate with optimizations
            with torch.no_grad():
                if use_fp16 and device == "cuda":
                    with torch.autocast(device_type="cuda"):
                        audio_values = model.generate(
                            **inputs, 
                            max_new_tokens=max_tokens,
                            guidance_scale=guidance_scale,
                            temperature=temperature,
                            do_sample=True,
                            num_beams=1,  # Faster than beam search
                            pad_token_id=processor.tokenizer.pad_token_id
                        )
                else:
                    audio_values = model.generate(
                        **inputs, 
                        max_new_tokens=max_tokens,
                        guidance_scale=guidance_scale,
                        temperature=temperature,
                        do_sample=True,
                        num_beams=1,
                        pad_token_id=processor.tokenizer.pad_token_id
                    )
            
            progress_bar.progress(80)
            status_text.text("💾 Saving audio...")
            
            # Save temp file
            sample_rate = model.config.audio_encoder.sampling_rate
            audio_data = audio_values[0, 0].cpu().numpy()
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                sf.write(tmp.name, audio_data, sample_rate)
                audio_path = tmp.name
            
            progress_bar.progress(100)
            status_text.text("✅ Generation complete!")
            
            # Clear some memory
            del audio_values
            if device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            
            st.success("🎉 Music generated successfully!")
            
            # Display audio player and download
            col1, col2 = st.columns([3, 1])
            with col1:
                st.audio(audio_path, format="audio/wav")
            with col2:
                with open(audio_path, "rb") as f:
                    st.download_button(
                        "⬇️ Download", 
                        f.read(), 
                        file_name=f"bgm_{prompt[:20].replace(' ', '_')}.wav", 
                        mime="audio/wav"
                    )
            
            # Show generation info
            with st.expander("Generation Details"):
                st.write(f"**Model:** {model_choice}")
                st.write(f"**Length:** {length} seconds")
                st.write(f"**Tokens:** {max_tokens}")
                st.write(f"**Sample Rate:** {sample_rate} Hz")
                st.write(f"**Device:** {device}")
                
        except Exception as e:
            st.error(f"Error during generation: {str(e)}")
            progress_bar.empty()
            status_text.empty()
        finally:
            # Cleanup
            progress_bar.empty()
            status_text.empty()

# Memory usage info
if device == "cuda":
    try:
        memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        st.sidebar.metric("GPU Memory", f"{memory_used:.1f}GB / {memory_total:.1f}GB")
    except:
        pass