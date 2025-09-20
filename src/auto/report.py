
import os, base64, io, soundfile as sf
import matplotlib.pyplot as plt
from jinja2 import Template

TEMPLATE = '''<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="utf-8"/>
<title>Relatório - {{ title }}</title>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<style>
body{font-family:system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, Arial, sans-serif; color:#0b1220; margin:40px;}
h1{margin-bottom:6px}
h2{margin:24px 0 8px}
.grid{display:grid; grid-template-columns:1fr; gap:18px;}
.card{border:1px solid #e5e7eb; border-radius:10px; padding:16px; box-shadow:0 4px 20px rgba(2,6,23,.04)}
.meta{font-size:14px; color:#475569}
img{max-width:100%; height:auto; border-radius:10px; border:1px solid #e5e7eb}
code{background:#f8fafc; padding:2px 6px; border-radius:6px; font-family:ui-monospace, SFMono-Regular, Menlo, Consolas, monospace}
.badge{display:inline-block; padding:2px 8px; border-radius:999px; background:#eef2ff; color:#3730a3; font-size:12px}
</style>
</head>
<body>
<h1>Relatório de Separação — {{ title }}</h1>
<div class="meta">Local: <code>{{ song_folder }}</code></div>
<h2>Stems gerados ({{ stems|length }})</h2>
<div class="grid">
{% for stem in stems %}
  <div class="card">
    <div class="badge">{{ stem.name }}</div>
    <p class="meta">{{ stem.path }}</p>
    <p class="meta">Provável: <strong>{{ stem.label }}</strong> — conf.: {{ "%.2f"|format(stem.confidence) }}</p>
    <img src="data:image/png;base64,{{ stem.wave }}" alt="waveform {{ stem.name }}"/>
  </div>
{% endfor %}
</div>
</body>
</html>'''

def _png_bytes(fig):
    bio = io.BytesIO()
    fig.savefig(bio, format="png", bbox_inches="tight", dpi=160)
    import matplotlib.pyplot as plt
    plt.close(fig)
    bio.seek(0)
    return base64.b64encode(bio.read()).decode("ascii")

def _waveform_png(path):
    data, sr = sf.read(path, always_2d=True)
    y = data.mean(axis=1)
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(os.path.basename(path))
    ax.plot(y)
    return _png_bytes(fig)

def build_report(song_folder, stems_meta, out_html, title):
    stems = []
    for m in stems_meta:
        try:
            png = _waveform_png(m["path"])
        except Exception:
            png = ""
        stems.append({
            "name": m["name"],
            "path": m["path"],
            "label": m["label"],
            "confidence": m.get("confidence", 0.0),
            "wave": png
        })
    html = Template(TEMPLATE).render(title=title, song_folder=song_folder, stems=stems)
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    return out_html
