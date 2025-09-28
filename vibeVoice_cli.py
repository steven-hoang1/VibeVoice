#!/usr/bin/env python3
"""
CLI wrapper for VibeVoice that bypasses Gradio interface
This directly uses the synthesis functions from gradio_demo.py
"""

import sys
import os
import argparse

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'demo'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--voice", type=str, required=True, help="Voice prompt audio file")
    parser.add_argument("--output", type=str, required=True, help="Output audio file")
    parser.add_argument("--model_path", type=str, default="microsoft/VibeVoice-1.5B")
    parser.add_argument("--speaker", type=str, default="Speaker")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # Import everything from gradio_demo
    import gradio_demo
    
    # Format text
    formatted_text = f"<|speaker|>{args.speaker}<|text|>{args.text}"
    
    # Look for the synthesis function
    # Common names: synthesize, generate, process, infer, predict
    synthesis_func = None
    for name in ['synthesize', 'generate', 'process', 'infer', 'predict', 'synthesize_speech']:
        if hasattr(gradio_demo, name):
            synthesis_func = getattr(gradio_demo, name)
            if callable(synthesis_func):
                print(f"Using function: {name}")
                break
    
    if not synthesis_func:
        # Try to find any function that might work
        for name in dir(gradio_demo):
            if not name.startswith('_'):
                obj = getattr(gradio_demo, name)
                if callable(obj) and 'text' in str(obj.__code__.co_varnames):
                    synthesis_func = obj
                    print(f"Using function: {name}")
                    break
    
    if synthesis_func:
        try:
            # Try to call the function
            result = synthesis_func(formatted_text, args.voice)
            
            # Save the result
            if result:
                import torch
                import torchaudio
                
                if isinstance(result, tuple):
                    audio, sr = result
                    torchaudio.save(args.output, audio, sr)
                
                print(f"Saved to {args.output}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Could not find synthesis function")

if __name__ == "__main__":
    main()
