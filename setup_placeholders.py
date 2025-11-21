#!/usr/bin/env python3
"""
Setup script to generate placeholder media files for testing the pipeline.
Creates a 5-second dummy video and a 5-second silent audio file if they don't exist.
"""

import os
import cv2
import wave
import numpy as np
from pathlib import Path

def create_dummy_video(output_path, duration=5, fps=30, width=640, height=480):
    """Create a 5-second blank video using OpenCV."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = int(duration * fps)
    for _ in range(total_frames):
        # Create a blank black frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        out.write(frame)
    
    out.release()
    print(f"Created dummy video: {output_path} ({duration}s, {fps}fps, {width}x{height})")

def create_dummy_audio(output_path, duration=5, sample_rate=44100):
    """Create a 5-second silent audio file using wave (WAV format)."""
    # If output path is MP3, create WAV first then note conversion needed
    output_path = Path(output_path)
    if output_path.suffix.lower() == '.mp3':
        wav_path = output_path.with_suffix('.wav')
        print(f"Note: Creating WAV format ({wav_path}) as MP3 requires additional libraries.")
        print("Whisper can process WAV files directly, or you can convert manually.")
        output_path = wav_path
    
    num_samples = int(duration * sample_rate)
    
    # Create silent audio data (zeros)
    audio_data = np.zeros(num_samples, dtype=np.int16)
    
    with wave.open(str(output_path), 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)   # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())
    
    print(f"Created dummy audio: {output_path} ({duration}s, {sample_rate}Hz, mono)")

def main():
    """Check if input files exist, create placeholders if they don't."""
    base_dir = Path(__file__).parent
    inputs_dir = base_dir / "inputs"
    inputs_dir.mkdir(exist_ok=True)
    
    video_path = inputs_dir / "input_video.mp4"
    audio_path = inputs_dir / "input_audio.mp3"
    
    # Check and create video if it doesn't exist
    if not video_path.exists():
        print(f"Video file not found: {video_path}")
        print("Creating placeholder video...")
        create_dummy_video(str(video_path))
    else:
        print(f"Video file already exists: {video_path}")
    
    # Check and create audio if it doesn't exist
    # Whisper can handle both MP3 and WAV, so we'll create WAV if MP3 doesn't exist
    audio_wav_path = inputs_dir / "input_audio.wav"
    if not audio_path.exists():
        if not audio_wav_path.exists():
            print(f"Audio file not found: {audio_path}")
            print("Creating placeholder audio...")
            create_dummy_audio(str(audio_path))  # Will create WAV if MP3 requested
        else:
            print(f"Audio file (WAV) already exists: {audio_wav_path}")
    else:
        print(f"Audio file already exists: {audio_path}")

if __name__ == "__main__":
    main()

