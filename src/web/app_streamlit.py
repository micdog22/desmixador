
import streamlit as st, os, tempfile, glob
from auto.utils import ensure_dir, basename_noext, detect_device, zip_dir, band_split
from auto.engine import run_demucs, find_output_folder
from auto.nmf_split import split_file_nmf
from auto.classify import tag_wav, best_label
from auto.post import trim_silence, loudness_normalize, save_wav, export_mp3
from auto.report import build_report
import soundfile as sf, numpy as np

st.set_page_config(page_title="Desmixador", layout="centered")
st.title("Desmixador — Stems adaptativos")

uploaded = st.file_uploader("Envie um arquivo de áudio", type=["mp3","wav","flac","ogg","m4a","aac"])

col1, col2 = st.columns(2)
with col1:
    base_model = st.selectbox("Modelo Demucs base", ["htdemucs","htdemucs_6s"], index=0)
    gpu = st.checkbox("Tentar GPU (CUDA)", value=False)
    shifts = st.slider("Shifts Demucs", 0, 4, 2)
    overlap = st.slider("Overlap", 0.0, 0.95, 0.25, step=0.05)
with col2:
    mp3 = st.checkbox("Exportar MP3", value=True)
    bitrate = st.slider("Bitrate MP3 (kbps)", 128, 320, 320, step=32)
    normalize = st.checkbox("Normalizar (-14 LUFS)", value=True)
    trim = st.checkbox("Trim de silêncio", value=True)
    max_extra = st.slider("Máx. componentes extras (Auto‑K)", 0, 12, 6, step=1)
    drum_split = st.checkbox("Split de bateria (low/mid/high)", value=False)

if uploaded and st.button("Separar"):
    ensure_dir("outputs"); ensure_dir("reports")
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as tmp:
        tmp.write(uploaded.getvalue())
        tmp_path = tmp.name
    song = os.path.splitext(uploaded.name)[0]

    device = detect_device(gpu)
    with st.spinner("Demucs base..."):
        run_demucs(tmp_path, out_dir="outputs", device=device, model=base_model, shifts=shifts, overlap=overlap, mp3=False)
    os.remove(tmp_path)
    folder = find_output_folder("outputs", base_model, song)

    stems_meta = []
    def _finalize(name, y, sr0):
        if trim: y = trim_silence(y, sr0)
        if normalize: y = loudness_normalize(y, sr0, -14.0)
        dest = os.path.join(folder, f"{name}.wav")
        save_wav(dest, y, sr0)
        try:
            scores = tag_wav(dest); label = best_label(scores); conf = max(scores.values())
        except Exception:
            label = "desconhecido"; conf = 0.0
        final = dest
        if mp3:
            final = export_mp3(dest, bitrate=bitrate)
            try: os.remove(dest)
            except: pass
        stems_meta.append({"name": name, "path": final, "label": label, "confidence": conf})

    def _load(path):
        try:
            y, sr0 = sf.read(path, always_2d=False)
            if hasattr(y, "ndim") and y.ndim > 1: y = y.mean(axis=1)
            return y, sr0
        except Exception:
            return None, None

    base_stems = {}
    for name in ["vocals","drums","bass","other","piano","guitar"]:
        cand = glob.glob(os.path.join(folder, f"{name}.*"))
        if cand:
            y, sr0 = _load(cand[0])
            if y is not None: base_stems[name] = (y, sr0)

    for n in ["vocals","bass"]:
        if n in base_stems:
            _finalize(n, base_stems[n][0], base_stems[n][1])

    if "drums" in base_stems and drum_split:
        y, sr0 = base_stems["drums"]
        low, mid, high = band_split(y, sr0)
        _finalize("drums_low", low, sr0)
        _finalize("drums_mid", mid, sr0)
        _finalize("drums_high", high, sr0)
    elif "drums" in base_stems:
        _finalize("drums", base_stems["drums"][0], base_stems["drums"][1])

    for n in ["piano","guitar","other"]:
        if n in base_stems and os.path.exists(os.path.join(folder, f"{n}.wav")):
            with st.spinner(f"NMF (Auto‑K) no residual {n}..."):
                comps, sr1, k, errs = split_file_nmf(os.path.join(folder, f"{n}.wav"), max_k=max_extra, sr_target=None)
                for i, yi in enumerate(comps, start=1):
                    _finalize(f"{n}_comp{i}", yi, sr1)

    html = os.path.join("reports", f"{song}.html")
    build_report(folder, stems_meta, html, song)

    st.success("Concluído.")
    zip_path = os.path.join("outputs", f"{song}-autostems.zip")
    zip_dir(folder, zip_path)
    with open(zip_path, "rb") as f:
        st.download_button("Baixar ZIP dos stems", data=f, file_name=os.path.basename(zip_path), mime="application/zip")
    st.markdown(f"[Abrir relatório HTML]({html})")
