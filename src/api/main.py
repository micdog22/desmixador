
import os, tempfile
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from auto.utils import ensure_dir, basename_noext, detect_device, zip_dir, band_split
from auto.engine import run_demucs, find_output_folder
from auto.nmf_split import split_file_nmf
from auto.classify import tag_wav, best_label
from auto.post import trim_silence, loudness_normalize, save_wav, export_mp3
from auto.report import build_report
import soundfile as sf, glob, numpy as np

app = FastAPI(title="Desmixador API", version="1.0.0")

@app.post("/separate")
async def separate(file: UploadFile = File(...),
                   base_model: str = Form("htdemucs"),
                   mp3: bool = Form(True),
                   bitrate: int = Form(320),
                   gpu: bool = Form(False),
                   shifts: int = Form(0),
                   overlap: float = Form(0.25),
                   normalize: bool = Form(True),
                   trim: bool = Form(True),
                   sr: int | None = Form(None),
                   max_extra: int = Form(6),
                   drum_split: bool = Form(False)):
    ensure_dir("outputs"); ensure_dir("reports")
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    song = os.path.splitext(file.filename)[0]
    try:
        device = detect_device(gpu)
        run_demucs(tmp_path, out_dir="outputs", device=device, model=base_model, shifts=shifts, overlap=overlap, mp3=False)
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
                comps, sr1, k, errs = split_file_nmf(os.path.join(folder, f"{n}.wav"), max_k=max_extra, sr_target=sr)
                for i, yi in enumerate(comps, start=1):
                    _finalize(f"{n}_comp{i}", yi, sr1)

        html = os.path.join("reports", f"{song}.html")
        build_report(folder, stems_meta, html, song)

        zip_path = os.path.join("outputs", f"{song}-autostems.zip")
        zip_dir(folder, zip_path)
        return FileResponse(zip_path, media_type="application/zip", filename=os.path.basename(zip_path))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        try: os.remove(tmp_path)
        except: pass
