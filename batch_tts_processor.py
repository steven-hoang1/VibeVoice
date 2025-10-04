#!/usr/bin/env python3
"""
VibeVoice Batch Text-to-Speech Processor
Speed Control Version 2: Using Audio Resampling Method
This method changes playback speed which VibeVoice interprets as speaking rate
"""

import os
import sys
import re
import time
import argparse
import json
import tempfile
from pathlib import Path
from typing import List
import numpy as np
import torch
import torchaudio
import soundfile as sf
import librosa
from scipy import signal

# Add demo directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'demo'))

# Import the actual VibeVoice components from your gradio_demo
from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from transformers import set_seed


# ============================================================================
# SPEED CONTROL FUNCTIONALITY - RESAMPLING METHOD
# ============================================================================

class VoiceSpeedController:
    """
    Controls speaking speed using sample rate manipulation.
    This approach changes the playback speed of the voice sample,
    which VibeVoice then mimics in the generated speech.
    """
    
    def __init__(self):
        self.temp_files = []
    
    def adjust_voice_speed_resample(self, audio_path, speed_factor=1.0):
        """
        Adjust voice speed by resampling (changes pitch and speed together).
        
        For slower speech (speed_factor < 1.0):
          - We resample DOWN, making the audio play slower
        For faster speech (speed_factor > 1.0):
          - We resample UP, making the audio play faster
          
        Args:
            audio_path: Path to original voice file
            speed_factor: 0.5-2.0 (0.8 = 20% slower, 1.2 = 20% faster)
        """
        if speed_factor == 1.0:
            return audio_path
        
        print(f"  Adjusting voice speed (resample method): {speed_factor}x", end=" ")
        
        # Load audio
        audio, sr = sf.read(audio_path)
        
        # Handle stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Calculate new sample rate
        # If we want slower speech (0.8x), we need to lower the sample rate
        # If we want faster speech (1.2x), we need to increase the sample rate
        new_sr = int(sr / speed_factor)
        
        # Resample the audio
        audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=new_sr)
        
        # Create temp file
        temp_fd, output_path = tempfile.mkstemp(suffix='.wav', prefix='speed_adj_')
        os.close(temp_fd)
        self.temp_files.append(output_path)
        
        # Save at the NEW sample rate
        # When VibeVoice reads this, it will interpret the timing
        sf.write(output_path, audio_resampled, new_sr, subtype='PCM_16')
        
        if speed_factor < 1.0:
            print(f"({(1-speed_factor)*100:.0f}% slower)")
        else:
            print(f"({(speed_factor-1)*100:.0f}% faster)")
        
        return output_path
    
    def cleanup(self):
        """Remove temporary files"""
        for temp_file in self.temp_files:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
        self.temp_files = []
    
    def __del__(self):
        self.cleanup()


# ============================================================================
# REST OF YOUR ORIGINAL CODE
# ============================================================================

def count_words(text: str) -> int:
    """Count words in text"""
    return len(text.split())


def split_text_into_chunks(text: str, max_words: int = 400) -> List[str]:
    """Split text into chunks preserving sentence boundaries"""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for sentence in sentences:
        word_count = len(sentence.split())
        
        if current_word_count + word_count > max_words and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_word_count = word_count
        else:
            current_chunk.append(sentence)
            current_word_count += word_count
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks


class VibeVoiceBatchProcessor:
    """Batch processor using the actual VibeVoice model from gradio_demo"""
    
    def __init__(self, model_path: str, device: str = None, inference_steps: int = 10):
        """Initialize the VibeVoice model"""
        
        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self.model_path = model_path
        self.device = device
        self.inference_steps = inference_steps
        
        print(f"Initializing VibeVoice...")
        print(f"  Model path: {model_path}")
        print(f"  Device: {device}")
        print(f"  Inference steps: {inference_steps}")
        
        self.load_model()
    
    def load_model(self):
        """Load the VibeVoice model and processor"""
        
        print(f"Loading processor & model from {self.model_path}")
        
        # Normalize device name
        if self.device.lower() == "mpx":
            self.device = "mps"
        if self.device == "mps" and not torch.backends.mps.is_available():
            print("Warning: MPS not available. Falling back to CPU.")
            self.device = "cpu"
        
        # Load processor
        self.processor = VibeVoiceProcessor.from_pretrained(self.model_path)
        
        # Decide dtype & attention
        if self.device == "mps":
            load_dtype = torch.float32
            attn_impl = "sdpa"
        elif self.device == "cuda":
            load_dtype = torch.bfloat16
            attn_impl = "sdpa"
        else:
            load_dtype = torch.float32
            attn_impl = "sdpa"
        
        print(f"Using torch_dtype: {load_dtype}, attn_implementation: {attn_impl}")
        
        # Load model
        try:
            if self.device == "mps":
                self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=load_dtype,
                    attn_implementation=attn_impl,
                    device_map=None,
                )
                self.model.to("mps")
            elif self.device == "cuda":
                self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=load_dtype,
                    device_map="cuda",
                    attn_implementation=attn_impl,
                )
            else:
                self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=load_dtype,
                    device_map="cpu",
                    attn_implementation=attn_impl,
                )
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Trying fallback configuration...")
            
            self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                self.model_path,
                torch_dtype=load_dtype,
                device_map=(self.device if self.device in ("cuda", "cpu") else None),
                attn_implementation="sdpa",
            )
            if self.device == "mps":
                self.model.to("mps")
        
        self.model.eval()
        
        # Configure noise scheduler
        self.model.model.noise_scheduler = self.model.model.noise_scheduler.from_config(
            self.model.model.noise_scheduler.config, 
            algorithm_type='sde-dpmsolver++',
            beta_schedule='squaredcos_cap_v2'
        )
        self.model.set_ddpm_inference_steps(num_steps=self.inference_steps)
        
        print("✓ Model loaded successfully")
    
    def read_audio(self, audio_path: str, target_sr: int = 24000) -> np.ndarray:
        """Read and preprocess audio file"""
        try:
            wav, sr = sf.read(audio_path)
            if len(wav.shape) > 1:
                wav = np.mean(wav, axis=1)
            if sr != target_sr:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
            return wav
        except Exception as e:
            print(f"Error reading audio {audio_path}: {e}")
            try:
                wav, sr = torchaudio.load(audio_path)
                wav = wav.numpy()
                if len(wav.shape) > 1:
                    wav = np.mean(wav, axis=0)
                if sr != target_sr:
                    wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
                return wav
            except:
                raise e
    
    def generate_audio(self, text: str, voice_path: str, speaker_name: str = "Speaker", 
                  cfg_scale: float = 1.3, seed: int = 42, add_padding: bool = True) -> tuple:
        """Generate audio for a text chunk"""
        
        set_seed(seed)
        
        # Load voice sample
        voice_audio = self.read_audio(voice_path)
        
        # Process text
        processed_text = text.strip()
        
        if processed_text and not processed_text[-1] in '.!?':
            processed_text += '.'
        
        if add_padding:
            padding_options = [
                " That's all.",
                " The end.",
                " Thank you.",
                " [pause]",
                " ..."
            ]
            import random
            padding = random.choice(padding_options)
            processed_text += " This sentence is just padding to help the model finish properly." + padding
        
        formatted_text = f"Speaker 0: {processed_text}"
            
        # Prepare inputs
        inputs = self.processor(
            text=[formatted_text],
            voice_samples=[[voice_audio]],
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        
        # Move to device
        target_device = self.device if self.device in ("cuda", "mps") else "cpu"
        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(target_device)
        
        # Generate audio
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=cfg_scale,
                tokenizer=self.processor.tokenizer,
                generation_config={
                    'do_sample': False,
                },
                verbose=False,
                refresh_negative=False,
            )
        
        # Extract audio
        generated_audio = None
        
        if hasattr(outputs, 'speech_outputs'):
            speech_outputs = outputs.speech_outputs
            
            if speech_outputs is not None:
                if isinstance(speech_outputs, (list, tuple)) and len(speech_outputs) > 0:
                    generated_audio = speech_outputs[0]
                else:
                    generated_audio = speech_outputs
                    
                if isinstance(generated_audio, (list, tuple)) and len(generated_audio) > 0:
                    generated_audio = generated_audio[0]
        
        if generated_audio is None:
            possible_attrs = ['audios', 'audio', 'audio_values', 'waveform', 'wav']
            
            for attr in possible_attrs:
                if hasattr(outputs, attr):
                    audio_data = getattr(outputs, attr)
                    if audio_data is not None:
                        if isinstance(audio_data, (list, tuple)) and len(audio_data) > 0:
                            generated_audio = audio_data[0]
                        else:
                            generated_audio = audio_data
                        break
        
        if generated_audio is None:
            raise ValueError("Could not extract audio from outputs")
        
        # Convert to numpy
        if torch.is_tensor(generated_audio):
            if len(generated_audio.shape) == 3:
                generated_audio = generated_audio.squeeze(0)
                if len(generated_audio.shape) == 2:
                    if generated_audio.shape[0] <= 2:
                        generated_audio = generated_audio[0]
                    else:
                        generated_audio = generated_audio[:, 0]
            elif len(generated_audio.shape) == 2:
                if generated_audio.shape[0] <= 2:
                    generated_audio = generated_audio[0]
                else:
                    generated_audio = generated_audio[:, 0]
            
            if generated_audio.dtype == torch.bfloat16:
                generated_audio = generated_audio.float()
            
            audio_np = generated_audio.cpu().numpy()
        else:
            audio_np = np.array(generated_audio)
        
        if len(audio_np.shape) > 1:
            audio_np = audio_np.squeeze()
        
        sample_rate = 24000
        
        return audio_np, sample_rate


def convert_to_16_bit_wav(data):
    """Convert audio data to 16-bit WAV format"""
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
    
    data = np.array(data)
    
    if np.max(np.abs(data)) > 1.0:
        data = data / np.max(np.abs(data))
    
    data = (data * 32767).astype(np.int16)
    return data


def main():
    parser = argparse.ArgumentParser(description="VibeVoice Batch TTS - Speed Control V2")
    parser.add_argument("--text_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="audio_outputs")
    parser.add_argument("--model_path", type=str, default="microsoft/VibeVoice-1.5B")
    parser.add_argument("--voice_prompt", type=str, required=True)
    parser.add_argument("--speaker_name", type=str, default="Speaker")
    parser.add_argument("--chunk_size", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg_scale", type=float, default=1.3)
    parser.add_argument("--inference_steps", type=int, default=10)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--fix_cutoff", action="store_true", default=True)
    parser.add_argument("--speed_factor", type=float, default=1.0, 
                       help="Speaking speed (0.5-2.0). <1.0=slower, >1.0=faster")
    
    args = parser.parse_args()
    
    # Validate speed
    if args.speed_factor < 0.5 or args.speed_factor > 2.0:
        print(f"Warning: Speed {args.speed_factor} outside 0.5-2.0 range, clamping...")
        args.speed_factor = max(0.5, min(2.0, args.speed_factor))
    
    print("="*70)
    print("VibeVoice Batch TTS - Speed Control V2 (Resampling Method)")
    print("="*70)
    
    if not os.path.exists(args.text_file):
        print(f"Error: Text file not found: {args.text_file}")
        sys.exit(1)
    
    if not os.path.exists(args.voice_prompt):
        print(f"Error: Voice file not found: {args.voice_prompt}")
        sys.exit(1)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Text: {args.text_file}")
    print(f"Voice: {args.voice_prompt}")
    print(f"Output: {args.output_dir}")
    print(f"Model: {args.model_path}")
    
    if args.speed_factor != 1.0:
        if args.speed_factor < 1.0:
            print(f"🎚️  Speed: {args.speed_factor}x ({(1-args.speed_factor)*100:.0f}% SLOWER)")
        else:
            print(f"🎚️  Speed: {args.speed_factor}x ({(args.speed_factor-1)*100:.0f}% FASTER)")
    else:
        print(f"🎚️  Speed: {args.speed_factor}x (NORMAL)")
    
    print("="*70)
    
    # Initialize speed controller
    speed_controller = VoiceSpeedController()
    
    # Adjust voice using resampling method
    voice_to_use = args.voice_prompt
    if args.speed_factor != 1.0:
        print(f"\n🎛️  Preparing speed-adjusted voice (resampling method)...")
        voice_to_use = speed_controller.adjust_voice_speed_resample(
            args.voice_prompt, 
            args.speed_factor
        )
        print(f"✓ Speed-adjusted voice ready")
    
    # Initialize processor
    print("\nInitializing VibeVoice...")
    processor = VibeVoiceBatchProcessor(
        model_path=args.model_path,
        device=args.device,
        inference_steps=args.inference_steps
    )
    
    # Read text
    with open(args.text_file, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    
    total_words = count_words(text)
    print(f"\nTotal words: {total_words:,}")
    
    # Split chunks
    chunks = split_text_into_chunks(text, args.chunk_size)
    print(f"Chunks: {len(chunks)}")
    
    if args.test:
        print("\nTEST MODE: First chunk only")
        chunks = chunks[:1]
    
    # Process
    print("\n" + "="*70)
    print("Processing...")
    print("="*70)
    
    successful = 0
    failed = 0
    all_files = []
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\n[Chunk {i}/{len(chunks)}] {count_words(chunk)} words")
        preview = chunk[:80] + "..." if len(chunk) > 80 else chunk
        print(f"Preview: {preview}")
        
        output_file = os.path.join(args.output_dir, f"chunk_{i:03d}.wav")
        
        try:
            start_time = time.time()
            
            audio_np, sr = processor.generate_audio(
                text=chunk,
                voice_path=voice_to_use,
                speaker_name=args.speaker_name,
                cfg_scale=args.cfg_scale,
                seed=args.seed + i,
                add_padding=args.fix_cutoff
            )
            
            audio_16bit = convert_to_16_bit_wav(audio_np)
            sf.write(output_file, audio_16bit, sr, subtype='PCM_16')
            
            elapsed = time.time() - start_time
            duration = len(audio_16bit) / sr
            
            print(f"✓ Success! Time: {elapsed:.1f}s, Duration: {duration:.1f}s")
            print(f"  Saved: {output_file}")
            
            successful += 1
            all_files.append(output_file)
            
        except Exception as e:
            print(f"✗ Error: {e}")
            failed += 1
    
    # Cleanup
    speed_controller.cleanup()
    
    # Summary
    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)
    print(f"✓ Successful: {successful}/{len(chunks)}")
    if failed > 0:
        print(f"✗ Failed: {failed}/{len(chunks)}")
    
    print(f"\nOutput: {os.path.abspath(args.output_dir)}")
    
    # Manifest
    manifest = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_file": args.text_file,
        "total_words": total_words,
        "chunks": len(chunks),
        "successful": successful,
        "failed": failed,
        "voice_prompt": args.voice_prompt,
        "speed_factor": args.speed_factor,
        "speed_method": "resample",
        "model": args.model_path,
        "files": [os.path.basename(f) for f in all_files]
    }
    
    manifest_file = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"Manifest: {manifest_file}")
    print("\n✅ Done!")


if __name__ == "__main__":
    main()