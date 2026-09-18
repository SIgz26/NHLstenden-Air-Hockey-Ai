import math
import os
import wave

output_path = os.path.join(os.path.dirname(__file__), 'hover.wav')

sample_rate = 22050
seconds = 0.10
frames = []

for i in range(int(sample_rate * seconds)):
    t = i / sample_rate
    frequency = 1200.0 + 600.0 * math.sin(2.0 * math.pi * 3.0 * t)
    envelope = max(0.0, 1.0 - (t / seconds) * 1.3)
    sample = math.sin(2.0 * math.pi * frequency * t) * envelope
    value = int(max(-1.0, min(1.0, sample)) * 32767)
    frames.append(value.to_bytes(2, byteorder='little', signed=True))

with wave.open(output_path, 'wb') as wav_file:
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)
    wav_file.setframerate(sample_rate)
    wav_file.writeframes(b''.join(frames))

print(f'created: {output_path}')
print(f'size: {os.path.getsize(output_path)} bytes')
