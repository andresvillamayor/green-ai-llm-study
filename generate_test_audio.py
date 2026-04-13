#!/usr/bin/env python3
"""Genera audio WAV de prueba para Experimento C (ASR)."""

import numpy as np
import soundfile as sf
import os

def generate_test_audio(output_path="assets/sample_audio.wav", duration_s=3.0, sample_rate=16000, frequency_hz=440):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * frequency_hz * t)
    sf.write(output_path, audio, sample_rate, subtype='PCM_16')
    print(f"✅ Audio generado: {output_path}")
    print(f"   - Sample rate: {sample_rate} Hz | Mono | ~{os.path.getsize(output_path)/1024:.1f} KB")

if __name__ == "__main__":
    generate_test_audio()
