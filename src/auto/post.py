
import os
import numpy as np
import librosa, soundfile as sf
from pydub import AudioSegment
import pyloudnorm as pyln

def save_wav(path, y, sr):
    if hasattr(y, "ndim") and y.ndim > 1:
        y = librosa.to_mono(y)
    sf.write(path, y, sr, subtype="PCM_16")

def trim_silence(y, sr, top_db=40.0):
    idx = librosa.effects.split(y, top_db=top_db)
    if len(idx) == 0:
        return y
    start = idx[0, 0]
    end = idx[-1, 1]
    return y[start:end]

def loudness_normalize(y, sr, target_lufs=-14.0):
    meter = pyln.Meter(sr)
    loud = meter.integrated_loudness(y.astype(np.float64))
    gain = target_lufs - loud
    factor = 10 ** (gain / 20)
    return (y * factor).astype(np.float32)

def export_mp3(wav_path, bitrate=320):
    audio = AudioSegment.from_file(wav_path)
    mp3_path = os.path.splitext(wav_path)[0] + ".mp3"
    audio.export(mp3_path, format="mp3", bitrate=f"{bitrate}k")
    return mp3_path
