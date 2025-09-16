import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import soundfile as sf
import time
import gc
import argparse

class OptimizedMusicGen:
    def __init__(self, model_name="facebook/musicgen-small", device=None, use_fp16=True):
        """Initialize optimized MusicGen pipeline"""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = use_fp16 and self.device == "cuda"
        
        print(f"🔄 Loading model: {model_name}")
        print(f"📱 Device: {self.device}")
        print(f"🔢 FP16: {self.use_fp16}")
        
        start_time = time.time()
        
        # Load model with optimizations
        self.model = MusicgenForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.use_fp16 else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        # Move to device and optimize
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # PyTorch 2.0+ optimization
        if self.device == "cuda":
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("✅ Model compiled with torch.compile")
            except Exception as e:
                print(f"⚠️ Could not compile model: {e}")
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.2f} seconds")
        
    def generate_music(self, prompt, length_seconds=15, guidance_scale=3.0, temperature=1.0):
        """Generate music with optimizations"""
        print(f"🎵 Generating music for: '{prompt[:50]}...'")
        
        # Calculate tokens
        tokens_per_second = 50
        max_tokens = int(length_seconds * tokens_per_second)
        
        start_time = time.time()
        
        # Process inputs
        inputs = self.processor(
            text=[prompt],
            padding=True,
            return_tensors="pt",
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate with optimizations
        with torch.no_grad():
            if self.use_fp16:
                with torch.autocast(device_type="cuda"):
                    audio_values = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        guidance_scale=guidance_scale,
                        temperature=temperature,
                        do_sample=True,
                        num_beams=1,  # Faster than beam search
                        pad_token_id=self.processor.tokenizer.pad_token_id
                    )
            else:
                audio_values = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    guidance_scale=guidance_scale,
                    temperature=temperature,
                    do_sample=True,
                    num_beams=1,
                    pad_token_id=self.processor.tokenizer.pad_token_id
                )
        
        generation_time = time.time() - start_time
        print(f"⏱️ Generation completed in {generation_time:.2f} seconds")
        
        # Extract audio data
        audio_data = audio_values[0, 0].cpu().numpy()
        sample_rate = self.model.config.audio_encoder.sampling_rate
        
        # Memory cleanup
        del audio_values
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        
        return audio_data, sample_rate, generation_time
    
    def save_audio(self, audio_data, sample_rate, filename):
        """Save audio to file"""
        sf.write(filename, audio_data, sample_rate)
        print(f"💾 Audio saved as {filename}")
    
    def get_memory_usage(self):
        """Get current memory usage"""
        if self.device == "cuda":
            memory_used = torch.cuda.memory_allocated() / 1024**3
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return f"{memory_used:.1f}GB / {memory_total:.1f}GB"
        return "CPU mode"

def main():
    parser = argparse.ArgumentParser(description="Generate music with optimized MusicGen")
    parser.add_argument("--model", default="facebook/musicgen-small", 
                       choices=["facebook/musicgen-small", "facebook/musicgen-medium"],
                       help="Model to use")
    parser.add_argument("--prompt", default="Calm and spiritual background music with bansuri flute and soft tabla, suitable for narration of Ramayana story.",
                       help="Text prompt for music generation")
    parser.add_argument("--length", type=int, default=15, help="Length in seconds")
    parser.add_argument("--output", default="bgm_ramayana.wav", help="Output filename")
    parser.add_argument("--guidance", type=float, default=3.0, help="Guidance scale")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature")
    parser.add_argument("--no-fp16", action="store_true", help="Disable FP16")
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = OptimizedMusicGen(
        model_name=args.model,
        use_fp16=not args.no_fp16
    )
    
    # Generate music
    print(f"🎼 Generating {args.length}s of music...")
    audio_data, sample_rate, gen_time = generator.generate_music(
        prompt=args.prompt,
        length_seconds=args.length,
        guidance_scale=args.guidance,
        temperature=args.temperature
    )
    
    # Save output
    generator.save_audio(audio_data, sample_rate, args.output)
    
    # Show stats
    print(f"📊 Stats:")
    print(f"   • Generation time: {gen_time:.2f}s")
    print(f"   • Audio length: {args.length}s")
    print(f"   • Real-time factor: {args.length/gen_time:.2f}x")
    print(f"   • Sample rate: {sample_rate}Hz")
    print(f"   • Memory usage: {generator.get_memory_usage()}")

if __name__ == "__main__":
    main()