from transformers import AutoProcessor, MusicgenForConditionalGeneration
import torch
import soundfile as sf

# Load the pre-trained MusicGen model
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")

# Example: Background score for narration
prompt = "Calm and spiritual background music with bansuri flute and soft tabla, suitable for narration of Ramayana story."

inputs = processor(
    text=[prompt],
    padding=True,
    return_tensors="pt",
)

with torch.no_grad():
    audio_values = model.generate(**inputs, max_new_tokens=512)  # ~20-30 sec

# Save output as WAV
sf.write("bgm_ramayana.wav", audio_values[0, 0].cpu().numpy(), model.config.audio_encoder.sampling_rate)

print("✅ BGM generated and saved as bgm_ramayana.wav")
