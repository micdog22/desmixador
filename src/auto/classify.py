
import numpy as np
from typing import Dict
from panns_inference import AudioTagging

INSTRUMENT_CATS = {
    "Electric guitar": "guitarra",
    "Acoustic guitar": "violão",
    "Piano": "piano",
    "Drum": "bateria",
    "Drum kit": "bateria",
    "Snare drum": "caixa",
    "Bass drum": "bumbo",
    "Cymbal": "pratos",
    "Hi-hat": "chimbal",
    "Bass guitar": "baixo",
    "Violin, fiddle": "cordas",
    "Cello": "cordas",
    "Trumpet": "metais",
    "Trombone": "metais",
    "Saxophone": "sopro",
    "Flute": "sopro",
    "Synthesizer": "sintetizador",
    "Vocal music": "voz",
    "Choir": "coro",
    "Male singing": "voz",
    "Female singing": "voz",
    "Opera": "voz",
}

def tag_wav(path: str) -> Dict[str, float]:
    at = AudioTagging(checkpoint_path=None, device="cpu")
    (clipwise_output, labels, _) = at.inference(path)
    scores = {labels[i]: float(clipwise_output[0][i]) for i in range(len(labels))}
    return scores

def best_label(scores: Dict[str, float]) -> str:
    best = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
    for k, v in best:
        if k in INSTRUMENT_CATS:
            return INSTRUMENT_CATS[k]
    return "desconhecido"
