#!/usr/bin/env python3
"""
Audio Transcription Module using OpenAI Whisper.
Transcribes audio file and exports to CSV with timestamped segments.
"""

import os
import whisper
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.io import wavfile
from scipy import signal


def transcribe_audio(audio_path, model_name="base", output_csv_path=None):
    """
    Transcribe audio file using Whisper and save to CSV.
    
    Args:
        audio_path: Path to input audio file (MP3 or WAV)
        model_name: Whisper model to use (default: "base")
        output_csv_path: Path to output CSV file
    
    Returns:
        DataFrame with transcription segments
    """
    # Load Whisper model
    print(f"Loading Whisper model: {model_name}...")
    model = whisper.load_model(model_name)
    
    # Load audio file
    audio_path = Path(audio_path)
    print(f"Loading audio: {audio_path}...")
    
    # Try to load WAV directly without ffmpeg
    if audio_path.suffix.lower() == '.wav':
        try:
            sample_rate, audio = wavfile.read(str(audio_path))
            # Convert to float32 and normalize to [-1, 1]
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            elif audio.dtype == np.int32:
                audio = audio.astype(np.float32) / 2147483648.0
            # If stereo, convert to mono
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Whisper expects audio at 16kHz, so resample if necessary
            target_sr = 16000
            if sample_rate != target_sr:
                num_samples = int(len(audio) * target_sr / sample_rate)
                audio = signal.resample(audio, num_samples)
                print(f"Resampled audio from {sample_rate}Hz to {target_sr}Hz")
            else:
                print(f"Loaded WAV: {sample_rate}Hz, {len(audio)/sample_rate:.2f}s")
            
            # Transcribe with audio array (now at 16kHz)
            result = model.transcribe(audio, language=None)
        except Exception as e:
            print(f"Warning: Could not load WAV directly ({e}), trying with Whisper's loader...")
            result = model.transcribe(str(audio_path))
    else:
        # For MP3 or other formats, use Whisper's loader (requires ffmpeg)
        result = model.transcribe(str(audio_path))
    
    # Extract segments with timestamps
    segments = []
    for segment in result["segments"]:
        segments.append({
            "start_time": segment["start"],
            "end_time": segment["end"],
            "text_content": segment["text"].strip()
        })
    
    # Create DataFrame with proper column structure
    if segments:
        df = pd.DataFrame(segments)
    else:
        # Create empty DataFrame with correct columns if no segments found
        df = pd.DataFrame(columns=["start_time", "end_time", "text_content"])
        print("Warning: No speech segments detected in audio (audio may be silent or contain no speech).")
    
    # Save to CSV
    if output_csv_path:
        output_csv_path = Path(output_csv_path)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv_path, index=False)
        print(f"Transcription saved to: {output_csv_path}")
        print(f"Total segments: {len(df)}")
    
    return df


def main():
    """Main function to run transcription."""
    # Set up paths
    base_dir = Path(__file__).parent.parent
    audio_path = base_dir / "inputs" / "input_audio.mp3"
    audio_wav_path = base_dir / "inputs" / "input_audio.wav"
    output_path = base_dir / "outputs" / "transcripts" / "transcription.csv"
    
    # Check if audio file exists (try MP3 first, then WAV)
    if audio_path.exists():
        input_audio = audio_path
    elif audio_wav_path.exists():
        input_audio = audio_wav_path
    else:
        raise FileNotFoundError(
            f"Audio file not found. Expected: {audio_path} or {audio_wav_path}\n"
            "Please run setup_placeholders.py first or provide your audio file."
        )
    
    # Transcribe
    transcribe_audio(input_audio, model_name="base", output_csv_path=output_path)


if __name__ == "__main__":
    main()

