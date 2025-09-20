
import os, numpy as np
import soundfile as sf
import librosa
from sklearn.decomposition import NMF

def stft_mag(y, n_fft=2048, hop_length=512):
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    return S, np.abs(S), np.angle(S)

def istft_mag(M, phase, hop_length=512):
    S = M * np.exp(1j*phase)
    y = librosa.istft(S, hop_length=hop_length)
    return y

def auto_k_nmf(mag, max_k=6, random_state=0):
    errs = []
    Ws, Hs = [], []
    for k in range(1, max_k+1):
        nmf = NMF(n_components=k, init="nndsvda", max_iter=400, random_state=random_state)
        W = nmf.fit_transform(mag)
        H = nmf.components_
        recon = W @ H
        err = np.linalg.norm(mag - recon) / (mag.size**0.5)
        errs.append(err)
        Ws.append(W); Hs.append(H)
    best_k = max_k
    for i in range(1, len(errs)):
        drop = (errs[i-1] - errs[i]) / max(errs[i-1], 1e-6)
        if drop < 0.05:
            best_k = i
            break
    return Ws[best_k-1], Hs[best_k-1], best_k, errs

def split_file_nmf(path_wav: str, max_k=6, sr_target=None):
    y, sr = librosa.load(path_wav, sr=sr_target, mono=True)
    S, M, P = stft_mag(y)
    W, H, k, errs = auto_k_nmf(M, max_k=max_k)
    comps = []
    recon = W @ H + 1e-9
    for i in range(k):
        Mi = (W[:, [i]] @ H[[i], :])
        mask = Mi / recon
        yi = istft_mag(mask * M, P)
        comps.append(yi.astype(np.float32))
    return comps, sr, k, errs
