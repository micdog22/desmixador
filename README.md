# Desmixador — Separação Adaptativa (Auto‑K) de Stems

[![CI](https://github.com/SEU_USUARIO/desmixador/actions/workflows/ci.yml/badge.svg)](https://github.com/SEU_USUARIO/desmixador/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Separação adaptativa de stems: detecta quantos vocais e instrumentos existem numa música e separa tudo que fizer sentido. Combina Demucs (separação base) + NMF Auto‑K (decomposição do residual) + classificação (PANNs/AudioSet) para rotular cada componente. Inclui CLI, API (FastAPI), Web UI (Streamlit), Docker e relatório HTML.

Nota: atualize `SEU_USUARIO` nas badges após publicar no GitHub.

---

## Instalação
Requisitos: Python 3.10+, FFmpeg. GPU é opcional (acelera).
```bash
pip install -r requirements.txt
```

## Uso rápido

### CLI
```bash
python src/autostems.py separate "inputs/minha_musica.mp3" -o outputs   --mp3 --bitrate 320 --normalize --trim --report --max-extra 6 --drum-split
```

### API
```bash
python src/autostems.py serve --host 0.0.0.0 --port 8000
# docs: http://localhost:8000/docs
```

### Web UI
```bash
python src/autostems.py webui
```

### Docker
```bash
docker build -t desmixador .
docker run --rm -v "$PWD/inputs:/app/inputs" -v "$PWD/outputs:/app/outputs" sonic-hydra   python src/autostems.py separate "inputs/minha_musica.mp3" -o outputs --report --mp3 --bitrate 320
```

---

## Como funciona (resumo)
1) Demucs (`htdemucs` por padrão) separa `vocals`, `drums`, `bass` e `other` (ou `htdemucs_6s` com `piano/guitar`).
2) NMF Auto‑K decompõe o residual (`other` e opcionalmente `piano/guitar`) em K componentes escolhidos automaticamente.
3) Classificação com PANNs atribui rótulos e confiança para cada componente.
4) Pós-processamento: -14 LUFS, trim de silêncio, MP3 320 kbps, relatório HTML.

## Limitações
Separar cada instrumento 100% isolado ainda é um desafio aberto. Este projeto entrega stems úteis e rotulados com transparência (relatório).

## Licença
MIT para este wrapper. Demucs e modelos de terceiros possuem licenças próprias.
