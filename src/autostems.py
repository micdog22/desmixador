
import os, pathlib, typer
from typing import Optional, List
from rich.console import Console
from rich.progress import track

from auto.utils import ensure_dir, scan_inputs, basename_noext, detect_device, zip_dir, band_split
from auto.engine import run_demucs, find_output_folder
from auto.nmf_split import split_file_nmf
from auto.classify import tag_wav, best_label
from auto.post import trim_silence, loudness_normalize, save_wav, export_mp3
from auto.report import build_report

app = typer.Typer(add_completion=False)
console = Console()

@app.command()
def separate(path: str = typer.Argument(..., help="Arquivo ou pasta"),
             out: str = typer.Option("outputs", "--out", "-o"),
             base_model: str = typer.Option("htdemucs", help="Modelo Demucs base: htdemucs (4) ou htdemucs_6s (6)"),
             mp3: bool = typer.Option(True, help="Exportar MP3"),
             bitrate: int = typer.Option(320, help="Bitrate MP3"),
             gpu: bool = typer.Option(False, help="Tentar GPU"),
             shifts: int = typer.Option(0, help="Shifts Demucs"),
             overlap: float = typer.Option(0.25, help="Overlap Demucs"),
             normalize: bool = typer.Option(True, help="Normalizar -14 LUFS"),
             trim: bool = typer.Option(True, help="Remover silêncio"),
             sr: Optional[int] = typer.Option(None, help="Reamostrar para SR"),
             max_extra: int = typer.Option(6, help="Máximo de componentes adicionais (Auto‑K)"),
             drum_split: bool = typer.Option(False, help="Dividir drums em bandas low/mid/high"),
             report: bool = typer.Option(True, help="Gerar relatório HTML")):
    files = scan_inputs(path)
    if not files: raise typer.BadParameter("Nenhum arquivo encontrado.")
    ensure_dir(out); ensure_dir("reports")

    device = detect_device(gpu)
    console.rule(f"[bold]Processando {len(files)} arquivo(s) | device={device} | modelo={base_model}")

    for f in track(files, description="Separando"):
        stem_base = basename_noext(f)
        run_demucs(f, out_dir=out, device=device, model=base_model, shifts=shifts, overlap=overlap, mp3=False)
        folder = find_output_folder(out, base_model, stem_base)

        stems_meta = []

        def _finalize(name, y, sr0):
            if trim: y = trim_silence(y, sr0)
            if normalize: y = loudness_normalize(y, sr0, -14.0)
            wav_path = os.path.join(folder, f"{name}.wav")
            save_wav(wav_path, y, sr0)
            try:
                scores = tag_wav(wav_path)
                label = best_label(scores)
                conf = max(scores.values()) if scores else 0.0
            except Exception:
                label = "desconhecido"; conf = 0.0
            final_path = wav_path
            if mp3:
                final_path = export_mp3(wav_path, bitrate=bitrate)
                try: os.remove(wav_path)
                except: pass
            stems_meta.append({"name": name, "path": final_path, "label": label, "confidence": conf})

        import soundfile as sf, glob, numpy as np
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
            target = os.path.join(folder, f"{n}.wav")
            if n in base_stems and os.path.exists(target):
                comps, sr1, k, errs = split_file_nmf(target, max_k=max_extra, sr_target=sr)
                for i, yi in enumerate(comps, start=1):
                    _finalize(f"{n}_comp{i}", yi, sr1)

        if report:
            html = os.path.join("reports", f"{stem_base}.html")
            build_report(folder, stems_meta, html, stem_base)
        zip_path = os.path.join(out, f"{stem_base}-autostems.zip")
        zip_dir(folder, zip_path)
        console.print(f"[green]OK:[/green] {zip_path}")

    console.rule("[bold green]Concluído")

@app.command()
def serve(host: str="127.0.0.1", port: int=8000):
    import uvicorn
    uvicorn.run("api.main:app", host=host, port=port, reload=False)

@app.command()
def webui():
    import subprocess, sys, pathlib
    app_path = pathlib.Path(__file__).parent / "web" / "app_streamlit.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])

if __name__ == "__main__":
    app()
