#!/usr/bin/env python3
"""
VibeVoice Batch Text-to-Speech Processor
Direct integration with your gradio_demo.py
"""

import os
import sys
import re
import time
import argparse
import json
from pathlib import Path
from typing import List
import numpy as np
import torch
import torchaudio
import soundfile as sf
import librosa

# Add demo directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'demo'))

# Import the actual VibeVoice components from your gradio_demo
from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from transformers import set_seed


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
        """Load the VibeVoice model and processor (from gradio_demo.py)"""
        
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
        
        # Configure noise scheduler (from gradio_demo)
        self.model.model.noise_scheduler = self.model.model.noise_scheduler.from_config(
            self.model.model.noise_scheduler.config, 
            algorithm_type='sde-dpmsolver++',
            beta_schedule='squaredcos_cap_v2'
        )
        self.model.set_ddpm_inference_steps(num_steps=self.inference_steps)
        
        print("✓ Model loaded successfully")
    
    def read_audio(self, audio_path: str, target_sr: int = 24000) -> np.ndarray:
        """Read and preprocess audio file (from gradio_demo)"""
        try:
            wav, sr = sf.read(audio_path)
            if len(wav.shape) > 1:
                wav = np.mean(wav, axis=1)
            if sr != target_sr:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
            return wav
        except Exception as e:
            print(f"Error reading audio {audio_path}: {e}")
            # Try with torchaudio as fallback
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
        
        # Set seed
        set_seed(seed)
        
        # Load voice sample
        voice_audio = self.read_audio(voice_path)
        
        # Process text to prevent last sentence cutoff
        processed_text = text.strip()
        
        # Strategy 1: Ensure proper ending punctuation
        if processed_text and not processed_text[-1] in '.!?':
            processed_text += '.'
        
        # Strategy 2: Add padding text (optional)
        if add_padding:
            # Add a simple padding sentence that won't be audible but helps the model complete
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
        
        # Format text with speaker
        formatted_text = f"Speaker 0: {processed_text}"
            
        # Prepare inputs (following gradio_demo structure)
        inputs = self.processor(
            text=[formatted_text],
            voice_samples=[[voice_audio]],  # Note: nested list for single speaker
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
        
        # Debug: Let's see what the output structure is
        print(f"Output type: {type(outputs)}")
        if hasattr(outputs, '__dict__'):
            print(f"Output attributes: {list(outputs.__dict__.keys())}")
        
        # Extract audio from outputs
        generated_audio = None
        
        # The audio should be in speech_outputs, not sequences!
        # sequences contains token IDs, speech_outputs contains the actual audio
        if hasattr(outputs, 'speech_outputs'):
            speech_outputs = outputs.speech_outputs
            print(f"Found speech_outputs, type: {type(speech_outputs)}")
            
            if speech_outputs is not None:
                # speech_outputs might be a list or tensor
                if isinstance(speech_outputs, (list, tuple)) and len(speech_outputs) > 0:
                    generated_audio = speech_outputs[0]
                else:
                    generated_audio = speech_outputs
                    
                # If it's still nested, extract further
                if isinstance(generated_audio, (list, tuple)) and len(generated_audio) > 0:
                    generated_audio = generated_audio[0]
                
                print(f"Extracted audio from speech_outputs")
                if hasattr(generated_audio, 'shape'):
                    print(f"Audio shape: {generated_audio.shape}")
                elif isinstance(generated_audio, (list, tuple)):
                    print(f"Audio is list/tuple with length: {len(generated_audio)}")
        
        # Fallback to other possible attributes if speech_outputs is None
        if generated_audio is None:
            possible_attrs = ['audios', 'audio', 'audio_values', 'waveform', 'wav']
            
            for attr in possible_attrs:
                if hasattr(outputs, attr):
                    audio_data = getattr(outputs, attr)
                    if audio_data is not None:
                        print(f"Found audio in attribute: {attr}")
                        if isinstance(audio_data, (list, tuple)) and len(audio_data) > 0:
                            generated_audio = audio_data[0]
                        else:
                            generated_audio = audio_data
                        break
        
        # Final check
        if generated_audio is None:
            # Print detailed debug info
            print(f"ERROR: Could not find audio in outputs!")
            print(f"Output type: {type(outputs)}")
            print(f"Output attributes: {dir(outputs)}")
            if hasattr(outputs, 'sequences'):
                seq = outputs.sequences
                print(f"sequences shape: {seq.shape if hasattr(seq, 'shape') else 'N/A'}, dtype: {seq.dtype if hasattr(seq, 'dtype') else 'N/A'}")
            if hasattr(outputs, 'speech_outputs'):
                sp = outputs.speech_outputs
                print(f"speech_outputs: {sp}")
            raise ValueError("Could not extract audio from outputs")
        
        # Convert to numpy
        if torch.is_tensor(generated_audio):
            # Handle different tensor dimensions
            if len(generated_audio.shape) == 3:
                # Shape might be [batch, channels, samples] or [batch, samples, channels]
                generated_audio = generated_audio.squeeze(0)  # Remove batch dimension
                if len(generated_audio.shape) == 2:
                    # If still 2D, take first channel or average
                    if generated_audio.shape[0] <= 2:  # Likely channels first
                        generated_audio = generated_audio[0]
                    else:  # Likely samples first
                        generated_audio = generated_audio[:, 0]
            elif len(generated_audio.shape) == 2:
                # Shape might be [channels, samples] or [samples, channels]
                if generated_audio.shape[0] <= 2:  # Likely channels first
                    generated_audio = generated_audio[0]
                else:  # Likely samples first
                    generated_audio = generated_audio[:, 0]
            
            # Convert dtype if needed
            if generated_audio.dtype == torch.bfloat16:
                generated_audio = generated_audio.float()
            elif generated_audio.dtype in [torch.int16, torch.int32, torch.int64]:
                # If it's integer type, it might be token IDs, not audio
                print(f"WARNING: Audio is integer type {generated_audio.dtype}, this might be token IDs not audio!")
                generated_audio = generated_audio.float()
            
            audio_np = generated_audio.cpu().numpy()
        else:
            audio_np = np.array(generated_audio)
        
        print(f"Final audio shape: {audio_np.shape}, dtype: {audio_np.dtype}, min: {audio_np.min():.3f}, max: {audio_np.max():.3f}")
        
        # Ensure 1D
        if len(audio_np.shape) > 1:
            audio_np = audio_np.squeeze()
        
        # Sample rate from VibeVoice is 24000
        sample_rate = 24000
        
        return audio_np, sample_rate


def convert_to_16_bit_wav(data):
    """Convert audio data to 16-bit WAV format (from gradio_demo)"""
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
    
    data = np.array(data)
    
    # Normalize to [-1, 1]
    if np.max(np.abs(data)) > 1.0:
        data = data / np.max(np.abs(data))
    
    # Scale to 16-bit
    data = (data * 32767).astype(np.int16)
    return data


def main():
    parser = argparse.ArgumentParser(description="VibeVoice Batch Text-to-Speech Processor")
    parser.add_argument("--text_file", type=str, required=True, help="Input text file")
    parser.add_argument("--output_dir", type=str, default="audio_outputs", help="Output directory")
    parser.add_argument("--model_path", type=str, default="microsoft/VibeVoice-1.5B", help="Model path")
    parser.add_argument("--voice_prompt", type=str, required=True, help="Voice audio file")
    parser.add_argument("--speaker_name", type=str, default="Speaker", help="Speaker name")
    parser.add_argument("--chunk_size", type=int, default=400, help="Words per chunk")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cfg_scale", type=float, default=1.3, help="CFG scale for generation")
    parser.add_argument("--inference_steps", type=int, default=10, help="Number of inference steps")
    parser.add_argument("--device", type=str, default=None, help="Device: cuda, mps, or cpu")
    parser.add_argument("--test", action="store_true", help="Test mode - first chunk only")
    parser.add_argument("--fix_cutoff", action="store_true", default=True, help="Apply fixes to prevent last sentence cutoff")
    parser.add_argument("--overlap_words", type=int, default=10, help="Number of words to overlap between chunks")
    
    args = parser.parse_args()
    
    print("="*70)
    print("VibeVoice Batch Text-to-Speech Processor")
    print("="*70)
    
    # Validate files
    if not os.path.exists(args.text_file):
        print(f"Error: Text file not found: {args.text_file}")
        sys.exit(1)
    
    if not os.path.exists(args.voice_prompt):
        print(f"Error: Voice file not found: {args.voice_prompt}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Text file: {args.text_file}")
    print(f"Voice: {args.voice_prompt}")
    print(f"Output: {args.output_dir}")
    print(f"Model: {args.model_path}")
    print(f"Speaker: {args.speaker_name}")
    print(f"Seed: {args.seed}")
    print(f"CFG Scale: {args.cfg_scale}")
    print(f"Chunk size: {args.chunk_size} words")
    print("="*70)
    
    # Initialize the processor
    print("\nInitializing VibeVoice model...")
    processor = VibeVoiceBatchProcessor(
        model_path=args.model_path,
        device=args.device,
        inference_steps=args.inference_steps
    )
    
    # Read text
    print("\nReading text file...")
    with open(args.text_file, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    
    total_words = count_words(text)
    print(f"Total words: {total_words:,}")
    
    # Split into chunks
    chunks = split_text_into_chunks(text, args.chunk_size)
    num_chunks = len(chunks)
    print(f"Split into {num_chunks} chunks of ~{args.chunk_size} words each")
    
    if args.test:
        print("\nTEST MODE: Processing only first chunk")
        chunks = chunks[:1]
    
    # Process chunks
    print("\n" + "="*70)
    print("Processing chunks...")
    print("="*70)
    
    successful = 0
    failed = 0
    all_audio_files = []
    unparsed_texts = []  # Store texts that couldn't be parsed
    
    # Process main chunks
    chunk_index = 1
    for chunk in chunks:
        chunk_words = count_words(chunk)
        print(f"\n[Chunk {chunk_index}/{len(chunks)}] {chunk_words} words")
        
        # Preview
        preview = chunk[:80] + "..." if len(chunk) > 80 else chunk
        print(f"Preview: {preview}")
        
        output_file = os.path.join(args.output_dir, f"chunk_{chunk_index:03d}.wav")
        
        try:
            print("Generating audio...")
            start_time = time.time()
            
            # Generate audio
            audio_np, sample_rate = processor.generate_audio(
                text=chunk,
                voice_path=args.voice_prompt,
                speaker_name=args.speaker_name,
                cfg_scale=args.cfg_scale,
                seed=args.seed + chunk_index,
                add_padding=args.fix_cutoff  # Use the fix_cutoff flag
            )
            
            # Convert to 16-bit WAV
            audio_16bit = convert_to_16_bit_wav(audio_np)
            
            # Save audio
            sf.write(output_file, audio_16bit, sample_rate, subtype='PCM_16')
            
            elapsed = time.time() - start_time
            duration = len(audio_16bit) / sample_rate
            file_size = os.path.getsize(output_file) / 1024
            
            print(f"✓ Success!")
            print(f"  Time: {elapsed:.1f}s")
            print(f"  Duration: {duration:.1f}s")
            print(f"  Size: {file_size:.1f} KB")
            print(f"  Saved: {output_file}")
            
            successful += 1
            all_audio_files.append(output_file)
            
        except Exception as e:
            error_msg = str(e)
            if "Could not parse line" in error_msg:
                # Extract the unparsed text from the error message
                print(f"⚠ Parse error detected. Saving for later processing...")
                unparsed_texts.append(chunk)
                print(f"  Added to unparsed queue (will be processed as chunk {len(chunks) + len(unparsed_texts)})")
            else:
                print(f"✗ Error: {e}")
                failed += 1
                
        chunk_index += 1
    
    # Process unparsed texts if any
    if unparsed_texts:
        print("\n" + "="*70)
        print(f"Processing {len(unparsed_texts)} unparsed chunks...")
        print("="*70)
        
        for i, unparsed_chunk in enumerate(unparsed_texts):
            chunk_num = len(chunks) + i + 1
            chunk_words = count_words(unparsed_chunk)
            print(f"\n[Unparsed Chunk {chunk_num}] {chunk_words} words")
            
            # Preview
            preview = unparsed_chunk[:80] + "..." if len(unparsed_chunk) > 80 else unparsed_chunk
            print(f"Preview: {preview}")
            
            output_file = os.path.join(args.output_dir, f"chunk_{chunk_num:03d}_unparsed.wav")
            
            # Try different formatting approaches
            formatting_attempts = [
                # Attempt 1: Clean the text by removing special characters
                lambda t: re.sub(r'[^\w\s\.\,\!\?\-]', ' ', t),
                # Attempt 2: Split into smaller sentences
                lambda t: '. '.join(t.split('.')[:5]),  # Take first 5 sentences
                # Attempt 3: Add explicit speaker tag
                lambda t: f"Speaker 0: {t[:400]}",  # Limit to 500 chars
                # Attempt 4: Remove all punctuation except periods
                lambda t: re.sub(r'[^\w\s\.]', '', t),
            ]
            
            success_attempt = False
            for attempt_num, formatter in enumerate(formatting_attempts, 1):
                try:
                    formatted_text = formatter(unparsed_chunk)
                    print(f"  Attempt {attempt_num}: Trying reformatted text...")
                    
                    start_time = time.time()
                    
                    # Generate audio with reformatted text
                    audio_np, sample_rate = processor.generate_audio(
                        text=formatted_text,
                        voice_path=args.voice_prompt,
                        speaker_name=args.speaker_name,
                        cfg_scale=args.cfg_scale,
                        seed=args.seed + chunk_num + attempt_num,
                        add_padding=args.fix_cutoff  # Add the padding parameter here too
                    )
                    
                    # Convert and save
                    audio_16bit = convert_to_16_bit_wav(audio_np)
                    sf.write(output_file, audio_16bit, sample_rate, subtype='PCM_16')
                    
                    elapsed = time.time() - start_time
                    duration = len(audio_16bit) / sample_rate
                    file_size = os.path.getsize(output_file) / 1024
                    
                    print(f"✓ Success on attempt {attempt_num}!")
                    print(f"  Time: {elapsed:.1f}s")
                    print(f"  Duration: {duration:.1f}s")
                    print(f"  Size: {file_size:.1f} KB")
                    print(f"  Saved: {output_file}")
                    
                    successful += 1
                    all_audio_files.append(output_file)
                    success_attempt = True
                    break
                    
                except Exception as e:
                    print(f"    Attempt {attempt_num} failed: {str(e)[:100]}")
                    continue
            
            if not success_attempt:
                print(f"✗ All attempts failed for unparsed chunk {chunk_num}")
                failed += 1
                
                # Save the problematic text to a file for manual inspection
                problem_file = os.path.join(args.output_dir, f"chunk_{chunk_num:03d}_failed.txt")
                with open(problem_file, 'w', encoding='utf-8') as f:
                    f.write(unparsed_chunk)
                print(f"  Saved problematic text to: {problem_file}")
    
    # Summary
    print("\n" + "="*70)
    print("PROCESSING COMPLETE")
    print("="*70)
    print(f"✓ Successful: {successful}/{len(chunks) + len(unparsed_texts)}")
    if failed > 0:
        print(f"✗ Failed: {failed}/{len(chunks) + len(unparsed_texts)}")
    
    if all_audio_files:
        print(f"\nGenerated {len(all_audio_files)} audio files:")
        for f in all_audio_files[:5]:
            print(f"  - {os.path.basename(f)}")
        if len(all_audio_files) > 5:
            print(f"  ... and {len(all_audio_files)-5} more")
    
    print(f"\nOutput directory: {os.path.abspath(args.output_dir)}")
    
    # Save manifest
    manifest_file = os.path.join(args.output_dir, "manifest.json")
    manifest = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_file": args.text_file,
        "total_words": total_words,
        "chunks": len(chunks),
        "unparsed_chunks": len(unparsed_texts),
        "successful": successful,
        "failed": failed,
        "voice_prompt": args.voice_prompt,
        "model": args.model_path,
        "seed": args.seed,
        "cfg_scale": args.cfg_scale,
        "files": [os.path.basename(f) for f in all_audio_files]
    }
    
    with open(manifest_file, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"Manifest saved: {manifest_file}")
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()