
import subprocess, sys, os, pathlib
from typing import Optional

def run_demucs(input_path: str, out_dir: str, device: str = "cpu",
               model: str = "htdemucs", shifts: int = 0, overlap: float = 0.25,
               mp3: bool = False, bitrate: int = 320):
    \"\"\"Executa Demucs para 4 stems (htdemucs) ou 6 stems (htdemucs_6s).\"\"\"
    cmd = [sys.executable, "-m", "demucs.separate",
           "-n", model,
           "--out", out_dir,
           "-d", device,
           "--overlap", str(overlap)]
    if shifts and shifts > 0:
        cmd += ["--shifts", str(shifts)]
    if mp3:
        cmd += ["--mp3", "--mp3-bitrate", str(bitrate)]
    cmd += [input_path]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if res.returncode != 0:
        raise RuntimeError(res.stdout)
    return res.stdout

def find_output_folder(out_dir: str, model: str, song_stem: str):
    root = os.path.join(out_dir, model)
    cand = os.path.join(root, song_stem)
    if os.path.isdir(cand):
        return cand
    for d in os.listdir(root):
        if d.startswith(song_stem):
            return os.path.join(root, d)
    return cand
