import os
import time
import subprocess
import requests

urls = [
    "https://upload.wikimedia.org/wikipedia/commons/b/ba/Jfk_rice_university_speech.ogg", # JFK Speech
    "https://upload.wikimedia.org/wikipedia/commons/3/34/Sound_Effect_-_Water_Drop.ogg", # Water Drop
    "https://upload.wikimedia.org/wikipedia/commons/d/d4/Trumpet_C4.ogg", # Trumpet
    "https://upload.wikimedia.org/wikipedia/commons/4/4b/The_Star-Spangled_Banner.ogg", # Anthem
    "https://upload.wikimedia.org/wikipedia/commons/c/c8/Example.ogg" # Accordion
]

os.makedirs("dataset/real_audio", exist_ok=True)
files = []

for i, url in enumerate(urls):
    name = url.split('/')[-1]
    raw_path = f"dataset/real_audio/raw_{name}"
    out_path = f"dataset/real_audio/track_{i}.wav"
    
    print(f"Downloading {name}...")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    try:
        r = requests.get(url, headers=headers, stream=True)
        r.raise_for_status()
        with open(raw_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                
        # Use ffmpeg to convert to 48kHz mono, and cut to 5 seconds to keep training fast
        subprocess.run(["ffmpeg", "-y", "-i", raw_path, "-ar", "48000", "-ac", "1", "-t", "5", out_path], check=True, capture_output=True)
        files.append(os.path.abspath(out_path))
        print(f"-> Saved 48kHz {out_path}")
        os.remove(raw_path)
    except Exception as e:
        print(f"Failed {name}: {e}")
    
    time.sleep(2) # Prevent 429 Too Many Requests

with open("dataset/sanity_filelist.txt", "w") as f:
    for file in files:
        f.write(file + "\n")

print(f"\nSuccessfully updated sanity_filelist.txt with {len(files)} files!")
