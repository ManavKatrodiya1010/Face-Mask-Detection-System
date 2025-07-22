# create_alert_wav.py
import wave, struct, math

duration = 0.5          # seconds
freq     = 1000         # Hz
rate     = 44100        # samples per second
amp      = 32767        # max amplitude for 16-bit audio

n_samples = int(duration * rate)
frames = bytearray()

for i in range(n_samples):
    t = i / rate
    sample = int(amp * math.sin(2 * math.pi * freq * t))
    frames.extend(struct.pack('<h', sample))

with wave.open("alert.wav", "w") as f:
    f.setnchannels(1)          # mono
    f.setsampwidth(2)          # 16‑bit
    f.setframerate(rate)
    f.writeframes(frames)

print("✅  alert.wav generated.")
