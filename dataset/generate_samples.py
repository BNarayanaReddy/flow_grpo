import torch
import torchaudio
import os

def generate_samples():
    os.makedirs('dataset/audio_samples', exist_ok=True)
    sr = 48000
    duration = 4 # 4 seconds

    # 1. Frequency Sweep (Chirp)
    t = torch.linspace(0, duration, int(sr * duration))
    f0 = 200
    f1 = 12000
    phase = 2 * torch.pi * (f0 * t + (f1 - f0) / (2 * duration) * t**2)
    sweep = torch.sin(phase).unsqueeze(0).to(torch.float32)
    torchaudio.save('dataset/audio_samples/sweep_48k.wav', sweep, sr)

    # 2. Harmonic Tones (Music-like)
    f_base = 440
    harmonics = torch.zeros_like(t)
    for i in range(1, 15): # up to 14th harmonic
        harmonics += (1.0 / i) * torch.sin(2 * torch.pi * (f_base * i) * t)
    harmonics = (harmonics / torch.max(torch.abs(harmonics))).unsqueeze(0).to(torch.float32)
    torchaudio.save('dataset/audio_samples/harmonics_48k.wav', harmonics, sr)

    # 3. White Noise (Testing full spectrum)
    noise = torch.randn(1, int(sr * duration)).to(torch.float32)
    noise = noise / torch.max(torch.abs(noise))
    torchaudio.save('dataset/audio_samples/noise_48k.wav', noise, sr)

    print("Successfully generated 48kHz audio files.")

if __name__ == '__main__':
    generate_samples()
