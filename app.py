import streamlit as st
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import soundfile as sf
import tempfile
import os
import gc
import time
import psutil
from threading import Timer

# Streamlit Cloud configuration
st.set_page_config(
    page_title="Text-to-Music BGM Generator", 
    layout="centered",
    initial_sidebar_state="expanded"
)

# Check if running on Streamlit Cloud
def is_streamlit_cloud():
    return os.environ.get('STREAMLIT_SHARING_MODE') == 'cloud' or 'streamlit.app' in os.environ.get('STREAMLIT_SERVER_HEADLESS', '')

IS_CLOUD = is_streamlit_cloud()

st.title("🎶 Text-to-Music BGM Generator")
if IS_CLOUD:
    st.markdown("🌐 **Running on Streamlit Cloud** - Powered by cloud CPUs!")
else:
    st.markdown("💻 **Running Locally** - Using your computer's resources")

# Cloud-specific warnings and info
if IS_CLOUD:
    st.info("""
    ☁️ **Streamlit Cloud Limits:**
    - ⏱️ Maximum 10-minute execution time
    - 💾 ~1GB RAM available  
    - 🎵 Recommended: 5-10 second audio clips
    - 🔄 App may restart if memory limit exceeded
    """)

# System resource monitoring
def get_system_resources():
    try:
        memory = psutil.virtual_memory()
        return {
            'total_memory': memory.total / (1024**3),
            'available_memory': memory.available / (1024**3),
            'memory_percent': memory.percent,
            'cpu_count': psutil.cpu_count()
        }
    except:
        return {'total_memory': 1.0, 'available_memory': 0.5, 'memory_percent': 50, 'cpu_count': 2}

resources = get_system_resources()

# Sidebar with resource info
with st.sidebar:
    st.header("System Resources")
    
    if IS_CLOUD:
        st.metric("Environment", "☁️ Streamlit Cloud")
        st.metric("CPU Cores", f"~{resources['cpu_count']}")
        st.metric("Available RAM", f"~{resources['available_memory']:.1f}GB")
    else:
        st.metric("Environment", "💻 Local")
        st.metric("CPU Cores", resources['cpu_count'])
        st.metric("Total RAM", f"{resources['total_memory']:.1f}GB")
        st.metric("Available RAM", f"{resources['available_memory']:.1f}GB")

    # Memory warning
    if resources['memory_percent'] > 80:
        st.error("⚠️ High memory usage detected!")

# Cloud-optimized settings
st.sidebar.header("Settings")

# Force small model for cloud deployment
model_choice = "facebook/musicgen-small"
if IS_CLOUD:
    st.sidebar.info("☁️ Cloud Mode: Using small model only")
else:
    st.sidebar.success("💻 Local Mode: Small model recommended")

# Shorter lengths for cloud
max_length = 10 if IS_CLOUD else 20
default_length = 5 if IS_CLOUD else 10

length = st.sidebar.slider(
    "Music length (seconds)", 
    min_value=3, 
    max_value=max_length,
    value=default_length,
    help="Shorter clips work best on Streamlit Cloud due to memory limits"
)

# Cloud-specific performance settings
with st.sidebar.expander("⚙️ Performance Settings"):
    if IS_CLOUD:
        st.info("Settings optimized automatically for cloud deployment")
        low_memory_mode = True
        aggressive_cleanup = True
    else:
        low_memory_mode = st.checkbox("Low Memory Mode", value=False)
        aggressive_cleanup = st.checkbox("Aggressive Cleanup", value=False)

# Timeout handler for cloud
class TimeoutHandler:
    def __init__(self, timeout_seconds=540):  # 9 minutes (before 10min limit)
        self.timeout_seconds = timeout_seconds
        self.timer = None
        
    def start_timeout(self):
        if IS_CLOUD:
            self.timer = Timer(self.timeout_seconds, self.timeout_callback)
            self.timer.start()
    
    def cancel_timeout(self):
        if self.timer:
            self.timer.cancel()
    
    def timeout_callback(self):
        st.error("⏰ Generation taking too long for cloud deployment. Try shorter audio length.")

# Model loading with cloud optimizations
@st.cache_resource(show_spinner=False)
def load_model_for_deployment(model_name, is_cloud_env):
    """Load model optimized for deployment environment"""
    try:
        loading_start = time.time()
        
        # Cloud-specific optimizations
        if is_cloud_env:
            # More aggressive memory management for cloud
            model = MusicgenForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                device_map="cpu",
                use_safetensors=True  # Faster loading
            )
        else:
            # Standard loading for local
            model = MusicgenForConditionalGeneration.from_pretrained(model_name)
        
        processor = AutoProcessor.from_pretrained(model_name)
        
        # CPU optimizations
        model = model.to("cpu")
        model.eval()
        
        # Memory cleanup after loading
        gc.collect()
        
        loading_time = time.time() - loading_start
        return model, processor, loading_time
        
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        if is_cloud_env:
            st.error("This might be due to Streamlit Cloud memory limits. Try refreshing the app.")
        return None, None, 0

# Initialize model with progress tracking
if 'model_loaded' not in st.session_state:
    with st.spinner("🔄 Loading MusicGen model..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        if IS_CLOUD:
            status_text.text("☁️ Loading on Streamlit Cloud servers...")
        else:
            status_text.text("💻 Loading on your local machine...")
        progress_bar.progress(25)
        
        model, processor, load_time = load_model_for_deployment(model_choice, IS_CLOUD)
        progress_bar.progress(100)
        
        if model is not None:
            st.session_state.model_loaded = True
            st.session_state.model = model
            st.session_state.processor = processor
            status_text.text(f"✅ Model loaded in {load_time:.1f}s")
        else:
            status_text.text("❌ Model loading failed")
            
        progress_bar.empty()
        time.sleep(1)
        status_text.empty()

if 'model_loaded' not in st.session_state:
    st.error("Failed to load model. Please refresh the page.")
    if IS_CLOUD:
        st.info("💡 **Cloud Tip:** If this keeps happening, Streamlit Cloud might be under high load. Try again in a few minutes.")
    st.stop()

# Success message
if IS_CLOUD:
    st.success("✅ Model loaded on Streamlit Cloud servers!")
else:
    st.success("✅ Model loaded locally!")

# Prompt selection optimized for quick generation
st.subheader("🎤 Music Prompt")

# Curated prompts that work well with short clips
quick_prompts = [
    "Gentle piano melody for relaxation",
    "Soft acoustic guitar with ambient pads", 
    "Calm flute music for meditation",
    "Uplifting orchestral strings",
    "Traditional sitar with tabla rhythm",
    "Ambient nature sounds with soft synth",
    "Peaceful harp and string ensemble"
]

prompt_selection = st.selectbox(
    "Choose a quick prompt or enter custom:",
    ["Custom"] + quick_prompts,
    index=1 if IS_CLOUD else 0
)

if prompt_selection == "Custom":
    prompt = st.text_area(
        "Enter your prompt:", 
        value="Gentle piano melody for relaxation",
        help="Be specific but concise for best results"
    )
else:
    prompt = prompt_selection

# Generation parameters
col1, col2 = st.columns(2)
with col1:
    guidance_scale = st.slider("Guidance Scale", 1.0, 4.0, 2.0 if IS_CLOUD else 3.0, 0.5)
with col2:
    temperature = st.slider("Temperature", 0.7, 1.3, 1.0, 0.1)

# Main generation button
if st.button("🎵 Generate Music", type="primary"):
    if not prompt or prompt.strip() == "":
        st.warning("Please select or enter a prompt.")
        st.stop()
    
    # Pre-generation checks
    current_memory = psutil.virtual_memory().percent if hasattr(psutil, 'virtual_memory') else 50
    
    if IS_CLOUD and current_memory > 85:
        st.error("🚨 Cloud memory limit reached. Please refresh the app and try again.")
        st.stop()
    
    # Timeout handler for cloud
    timeout_handler = TimeoutHandler()
    timeout_handler.start_timeout()
    
    # Calculate generation parameters
    tokens_per_second = 35 if IS_CLOUD else 40  # Slightly lower for cloud
    max_tokens = int(length * tokens_per_second)
    
    # Estimated time
    if IS_CLOUD:
        estimated_time = length * 8  # Faster cloud CPUs
        st.info(f"☁️ **Cloud Generation:** Estimated time ~{estimated_time//60:.0f}m {estimated_time%60:.0f}s")
    else:
        estimated_time = length * 15
        st.info(f"💻 **Local Generation:** Estimated time ~{estimated_time//60:.0f}m {estimated_time%60:.0f}s")
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        start_time = time.time()
        
        progress_bar.progress(10)
        status_text.text("🔄 Processing prompt...")
        
        # Process inputs
        inputs = st.session_state.processor(
            text=[prompt],
            padding=True,
            return_tensors="pt",
        )
        
        progress_bar.progress(30)
        status_text.text("🎼 Generating music...")
        
        # Generate with environment-specific optimizations
        with torch.no_grad():
            audio_values = st.session_state.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                guidance_scale=guidance_scale,
                temperature=temperature,
                do_sample=True,
                num_beams=1,
                pad_token_id=st.session_state.processor.tokenizer.pad_token_id,
                use_cache=True
            )
        
        progress_bar.progress(85)
        status_text.text("💾 Processing audio...")
        
        # Extract and save audio
        sample_rate = st.session_state.model.config.audio_encoder.sampling_rate
        audio_data = audio_values[0, 0].cpu().numpy()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            sf.write(tmp.name, audio_data, sample_rate)
            audio_path = tmp.name
        
        generation_time = time.time() - start_time
        
        # Aggressive cleanup for cloud
        if IS_CLOUD or aggressive_cleanup:
            del audio_values, inputs
            gc.collect()
        
        progress_bar.progress(100)
        timeout_handler.cancel_timeout()
        
        # Success display
        rt_factor = length / generation_time
        st.success(f"🎉 Generated {length}s of music in {generation_time:.1f}s ({rt_factor:.1f}x realtime)!")
        
        # Audio player and download
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.audio(audio_path, format="audio/wav")
        with col2:
            with open(audio_path, "rb") as f:
                st.download_button(
                    "⬇️ Download",
                    f.read(),
                    file_name=f"music_{length}s.wav",
                    mime="audio/wav"
                )
        with col3:
            st.metric("Speed", f"{rt_factor:.1f}x RT")
        
        # Stats
        with st.expander("📊 Generation Stats"):
            st.write(f"**Environment:** {'☁️ Streamlit Cloud' if IS_CLOUD else '💻 Local'}")
            st.write(f"**Audio Length:** {length} seconds")
            st.write(f"**Generation Time:** {generation_time:.1f} seconds")
            st.write(f"**Realtime Factor:** {rt_factor:.1f}x")
            st.write(f"**Sample Rate:** {sample_rate} Hz")
            
    except Exception as e:
        timeout_handler.cancel_timeout()
        st.error(f"❌ Generation failed: {str(e)}")
        
        if IS_CLOUD:
            st.markdown("""
            ### ☁️ Cloud Troubleshooting:
            - **Memory limit exceeded**: Refresh the app and try shorter audio
            - **Timeout**: Use 5-second clips for faster generation  
            - **High load**: Try again in a few minutes
            """)
        else:
            st.markdown("""
            ### 💻 Local Troubleshooting:
            - Close other applications to free memory
            - Try shorter audio lengths
            - Restart the app if issues persist
            """)
    
    finally:
        progress_bar.empty()
        status_text.empty()

# Deployment-specific tips
if IS_CLOUD:
    with st.expander("☁️ Streamlit Cloud Tips"):
        st.markdown("""
        **Optimized for cloud deployment:**
        
        ✅ **Best practices:**
        - Keep audio length 5-10 seconds for reliable generation
        - Use the suggested prompts for faster results
        - If the app restarts, it's likely due to memory limits
        
        ⚡ **Performance on Streamlit Cloud:**
        - ~2x faster CPUs than typical laptop
        - Limited to ~1GB RAM 
        - 10-minute maximum execution time
        - May restart under high memory usage
        
        🔄 **If generation fails:**
        1. Refresh the page
        2. Try a shorter audio length  
        3. Wait a few minutes if servers are busy
        """)
else:
    with st.expander("💻 Local Deployment Tips"):
        st.markdown("""
        **Running on your machine:**
        
        ✅ **Advantages:**
        - No time limits
        - More available RAM
        - Consistent performance
        
        🎯 **Optimization tips:**
        - Close unnecessary applications
        - Use shorter clips for faster results
        - Consider upgrading to a machine with more RAM/faster CPU
        """)