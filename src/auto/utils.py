
import os, zipfile, pathlib, shutil
from typing import List, Tuple
import numpy as np

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def is_audio(path: str) -> bool:
    return pathlib.Path(path).suffix.lower() in AUDIO_EXTS

def scan_inputs(path: str) -> List[str]:
    p = pathlib.Path(path)
    if p.is_file():
        if is_audio(str(p)):
            return [str(p.resolve())]
        raise ValueError(f"Arquivo não suportado: {p.suffix}")
    return [str(f.resolve()) for f in p.rglob("*") if is_audio(str(f))]

def basename_noext(path: str) -> str:
    return pathlib.Path(path).stem

def zip_dir(folder: str, out_zip: str) -> str:
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(folder):
            for f in files:
                full = os.path.join(root, f)
                rel = os.path.relpath(full, start=os.path.dirname(folder))
                z.write(full, arcname=rel)
    return out_zip

def detect_device(force_gpu: bool) -> str:
    try:
        import torch
        import torch.cuda
        if force_gpu and torch.cuda.is_available():
            return "cuda"
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

def band_split(y: np.ndarray, sr: int, low_hz=160, high_hz=6000):
    \"\"\"3 bandas: low (<low_hz), mid (low_hz..high_hz), high (>high_hz) via FFT mask simples.\"\"\"
    import numpy as np
    Y = np.fft.rfft(y, axis=-1)
    freqs = np.fft.rfftfreq(y.shape[-1], 1/sr)
    low_mask = freqs < low_hz
    high_mask = freqs > high_hz
    mid_mask = ~(low_mask | high_mask)
    def recon(mask):
        Ym = Y * mask
        ym = np.fft.irfft(Ym, axis=-1).astype(np.float32)
        return ym
    return recon(low_mask), recon(mid_mask), recon(high_mask)
