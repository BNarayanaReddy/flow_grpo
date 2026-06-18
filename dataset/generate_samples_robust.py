import torch
import torchaudio
import os
import math

def generate_samples():
    os.makedirs('dataset/real_audio', exist_ok=True)
    sr = 48000
    duration = 3 # 3 seconds each
    files = []

    t = torch.linspace(0, duration, int(sr * duration))

    # 1. Frequency Sweep (Chirp)
    f0, f1 = 200, 12000
    phase = 2 * torch.pi * (f0 * t + (f1 - f0) / (2 * duration) * t**2)
    sweep = torch.sin(phase).unsqueeze(0).to(torch.float32)
    torchaudio.save('dataset/real_audio/generated_sweep.wav', sweep, sr)
    files.append(os.path.abspath('dataset/real_audio/generated_sweep.wav'))

    # 2. Harmonic Tones (Music-like)
    f_base = 440
    harmonics = torch.zeros_like(t)
    for i in range(1, 15):
        harmonics += (1.0 / i) * torch.sin(2 * torch.pi * (f_base * i) * t)
    harmonics = (harmonics / torch.max(torch.abs(harmonics))).unsqueeze(0).to(torch.float32)
    torchaudio.save('dataset/real_audio/generated_harmonics.wav', harmonics, sr)
    files.append(os.path.abspath('dataset/real_audio/generated_harmonics.wav'))

    # 3. Pulsing White Noise (Testing transients/drums)
    noise = torch.randn(1, int(sr * duration)).to(torch.float32)
    envelope = torch.sin(2 * torch.pi * 2 * t).abs().unsqueeze(0) # 2 Hz pulse
    noise = (noise * envelope) / torch.max(torch.abs(noise))
    torchaudio.save('dataset/real_audio/generated_pulse_noise.wav', noise, sr)
    files.append(os.path.abspath('dataset/real_audio/generated_pulse_noise.wav'))

    # 4. Low Frequency Bass Drone
    bass = torch.sin(2 * torch.pi * 50 * t) + 0.5 * torch.sin(2 * torch.pi * 100 * t)
    bass = bass.unsqueeze(0).to(torch.float32)
    torchaudio.save('dataset/real_audio/generated_bass.wav', bass, sr)
    files.append(os.path.abspath('dataset/real_audio/generated_bass.wav'))

    # 5. Dual-Tone Alarm
    alarm = torch.sin(2 * torch.pi * 800 * t) * (torch.sin(2 * torch.pi * 4 * t) > 0).float()
    alarm += torch.sin(2 * torch.pi * 1000 * t) * (torch.sin(2 * torch.pi * 4 * t) <= 0).float()
    alarm = alarm.unsqueeze(0).to(torch.float32)
    torchaudio.save('dataset/real_audio/generated_alarm.wav', alarm, sr)
    files.append(os.path.abspath('dataset/real_audio/generated_alarm.wav'))

    with open("dataset/sanity_filelist.txt", "w") as f:
        for file in files:
            f.write(file + "\n")

    print(f"Successfully generated {len(files)} pristine 48kHz test tracks and updated sanity_filelist.txt!")

if __name__ == '__main__':
    generate_samples()
