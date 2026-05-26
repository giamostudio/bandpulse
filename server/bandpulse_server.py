#!/usr/bin/env python3
"""
BANDPULSE Server — local BPM, Key, Energy & Genre analyzer for Bandcamp
Run with: ~/lupin_env/bin/python ~/Desktop/BANDPULSE/server/bandpulse_server.py
"""
from flask import Flask, request, jsonify, Response
import tempfile, os, requests as req_lib, threading, json, base64

app = Flask(__name__)
_lock = threading.Lock()
_es = None
MODELS_DIR = os.path.expanduser("~/bandpulse_models")

# Detect engine at startup
try:
    import essentia.standard as _essentia_test
    USE_ESSENTIA = True
    print("  Engine: Essentia (full analysis)")
except Exception:
    USE_ESSENTIA = False
    print("  Engine: librosa (BPM + Key)")

def get_es():
    global _es
    if _es is None:
        import essentia.standard as es
        _es = es
    return _es

PITCH_CLASSES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
MAJOR_PROFILE = [6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88]
MINOR_PROFILE = [6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17]

def librosa_analyze(path):
    import librosa, numpy as np
    y, sr = librosa.load(path, sr=None, mono=True)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = round(float(np.atleast_1d(tempo)[0]), 1)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    best_score, best_key = -1, "C major"
    for i in range(12):
        r = np.roll(chroma_mean, -i)
        maj = float(np.corrcoef(r, MAJOR_PROFILE)[0,1])
        mn  = float(np.corrcoef(r, MINOR_PROFILE)[0,1])
        if maj > best_score: best_score, best_key = maj, f"{PITCH_CLASSES[i]} major"
        if mn  > best_score: best_score, best_key = mn,  f"{PITCH_CLASSES[i]} minor"
    rms = float(np.sqrt(np.mean(y**2)))
    energy = min(100, round(rms * 3000))
    return bpm, best_key, energy

def analyze_audio(path):
    if not USE_ESSENTIA:
        bpm, key_str, energy = librosa_analyze(path)
        scale = "major" if "major" in key_str else "minor"
        key = key_str.replace(" major","").replace(" minor","")
        return {"bpm": bpm, "key": key_str, "camelot": to_camelot(key_str),
                "energy": energy, "danceability": None, "genre": None, "genre_discogs": None}

    es = get_es()
    audio_44k = es.MonoLoader(filename=path, sampleRate=44100)()
    audio_16k = es.MonoLoader(filename=path, sampleRate=16000)()

    bpm, _, _, _, _ = es.RhythmExtractor2013(method="multifeature")(audio_44k)
    key, scale, key_strength = es.KeyExtractor()(audio_44k)
    rms = float(es.RMS()(audio_44k))
    energy_pct = min(100, round(rms * 3000))

    # Danceability
    try:
        dance_model = es.TensorflowPredictMusiCNN(
            graphFilename=os.path.join(MODELS_DIR, "danceability-musicnn-msd-1.pb"),
            output="model/Sigmoid"
        )
        dance_acts = dance_model(audio_16k)
        import numpy as np
        dance_score = round(float(np.mean(dance_acts)) * 100)
    except Exception:
        dance_score = None

    # Genre via Discogs effnet
    genre_label = None
    genre_discogs = None
    try:
        import numpy as np
        effnet = es.TensorflowPredictEffnetDiscogs(
            graphFilename=os.path.join(MODELS_DIR, "discogs-effnet-bs64-1.pb"),
            output="PartitionedCall:0"
        )
        with open(os.path.join(MODELS_DIR, "discogs-effnet-bs64-1.json")) as f:
            discogs_classes = json.load(f)["classes"]
        activations = effnet(audio_16k)
        probs = np.mean(activations, axis=0)
        top_idx = np.argsort(probs)[::-1][:5]
        top_genres = [(discogs_classes[i], float(probs[i])) for i in top_idx]
        genre_discogs = top_genres[0][0].split("---")[-1] if top_genres else None
        genre_label = get_dj_genre(top_genres)
    except Exception:
        pass

    return {
        "bpm": round(float(bpm), 1),
        "key": f"{key} {scale}",
        "camelot": to_camelot(f"{key} {scale}"),
        "energy": energy_pct,
        "danceability": dance_score,
        "genre": genre_label,
        "genre_discogs": genre_discogs,
    }

DJ_MAP = {
    "House": [
        "Electronic---House", "Electronic---Deep House", "Electronic---Tech House",
        "Electronic---Acid House", "Electronic---Garage House", "Electronic---Ghetto House",
        "Electronic---Electro House", "Electronic---Euro House", "Electronic---Chicago House",
    ],
    "Techno": [
        "Electronic---Techno", "Electronic---Deep Techno", "Electronic---Dub Techno",
        "Electronic---Industrial", "Electronic---EBM", "Electronic---Gabber",
    ],
    "Funk / Soul": [
        "Funk / Soul---Funk", "Funk / Soul---Soul", "Funk / Soul---Boogie",
        "Electronic---Disco", "Electronic---Nu-Disco", "Electronic---Broken Beat",
        "Jazz---Jazz-Funk", "Jazz---Soul-Jazz", "Jazz---Fusion",
        "Electronic---Future Jazz", "Electronic---Electro", "Electronic---Acid Jazz",
        "Funk / Soul---P.Funk", "Funk / Soul---Rhythm & Blues",
    ],
    "Ambient": [
        "Electronic---Ambient", "Electronic---Dark Ambient", "Electronic---Drone",
        "Electronic---Downtempo", "Electronic---Chillwave", "Electronic---Berlin-School",
        "Electronic---Experimental", "Electronic---Abstract",
    ],
    "Breaks": [
        "Electronic---Breakbeat", "Electronic---Breaks", "Electronic---Big Beat",
        "Electronic---Drum n Bass", "Electronic---Jungle", "Electronic---Breakcore",
    ],
    "Trance": [
        "Electronic---Trance", "Electronic---Goa Trance", "Electronic---Psy-Trance",
        "Electronic---Progressive Trance",
    ],
    "Hip-Hop": [
        "Hip Hop---Hip-Hop", "Hip Hop---Boom Bap", "Hip Hop---Instrumental",
        "Hip Hop---Gangsta", "Hip Hop---Abstract",
    ],
    "Jazz": [
        "Jazz---Contemporary Jazz", "Jazz---Free Jazz", "Jazz---Modern Jazz",
        "Jazz---Vocal", "Jazz---Bop", "Jazz---Hard Bop",
    ],
    "Electronic": [
        "Electronic---Synth-pop", "Electronic---Electropop", "Electronic---Synthwave",
        "Electronic---Industrial", "Electronic---Noise",
    ],
    "Classical": ["Classical---"],
    "Reggae": ["Reggae---", "Reggae---Dub", "Reggae---Roots Reggae"],
}

def get_dj_genre(top_genres):
    for label, _ in top_genres:
        for cat, tags in DJ_MAP.items():
            if any(label.startswith(t.rstrip("-")) for t in tags):
                return cat
    if top_genres:
        parent = top_genres[0][0].split("---")[0]
        return parent.strip() or "Other"
    return "Other"

CAMELOT = {
    "B major": "1B",  "G# minor": "1A",  "Ab minor": "1A",
    "F# major": "2B", "Gb major": "2B",  "D# minor": "2A", "Eb minor": "2A",
    "C# major": "3B", "Db major": "3B",  "A# minor": "3A", "Bb minor": "3A",
    "Ab major": "4B", "G# major": "4B",  "F minor": "4A",
    "Eb major": "5B", "D# major": "5B",  "C minor": "5A",
    "Bb major": "6B", "A# major": "6B",  "G minor": "6A",
    "F major": "7B",                     "D minor": "7A",
    "C major": "8B",                     "A minor": "8A",
    "G major": "9B",                     "E minor": "9A",
    "D major": "10B",                    "B minor": "10A",
    "A major": "11B",                    "F# minor": "11A", "Gb minor": "11A",
    "E major": "12B",                    "C# minor": "12A", "Db minor": "12A",
}

def to_camelot(key_str):
    return CAMELOT.get(key_str.strip(), "?")

@app.after_request
def cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response

@app.route("/ping")
def ping():
    return jsonify({"ok": True, "version": "1.0.0"})

@app.route("/analyze")
def analyze():
    audio_url = request.args.get("url")
    if not audio_url:
        return jsonify({"error": "missing url"}), 400
    tmp_path = None
    try:
        with _lock:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp_path = tmp.name
            r = req_lib.get(audio_url, timeout=30, stream=True,
                            headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(65536):
                    f.write(chunk)
            result = analyze_audio(tmp_path)
            os.unlink(tmp_path)
        return jsonify(result)
    except Exception as e:
        if tmp_path:
            try: os.unlink(tmp_path)
            except: pass
        return jsonify({"error": str(e)}), 500

@app.route("/album")
def album():
    raw = request.args.get("data", "")
    artist = request.args.get("artist", "Album")

    # Fix base64url padding
    pad = 4 - len(raw) % 4
    if pad != 4:
        raw += "=" * pad

    try:
        tracks = json.loads(base64.urlsafe_b64decode(raw).decode())
    except Exception:
        return Response("Invalid data", status=400)

    tracks_json = json.dumps(tracks)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BANDPULSE — {artist}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;900&display=swap');

  :root {{
    --bg: #0A0A0A;
    --surface: #111111;
    --border: #1E1E1E;
    --border-light: #2A2A2A;
    --text: #F0EDE8;
    --muted: #555;
    --accent: #E8FF47;
    --accent-dim: rgba(232,255,71,0.12);
    --red: #FF4D4D;
    --green: #47FF8A;
  }}

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'Inter', -apple-system, sans-serif;
    font-size: 13px;
    line-height: 1.5;
    min-height: 100vh;
  }}

  header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 20px 28px;
    border-bottom: 1px solid var(--border);
  }}

  .logo {{
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--accent);
  }}

  .artist-name {{
    font-size: 13px;
    color: var(--muted);
    max-width: 300px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }}

  .analyze-all {{
    background: var(--accent);
    color: #000;
    border: none;
    padding: 8px 20px;
    border-radius: 4px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    cursor: pointer;
    transition: opacity 0.2s;
  }}
  .analyze-all:hover {{ opacity: 0.85; }}
  .analyze-all:disabled {{ opacity: 0.3; cursor: default; }}

  table {{
    width: 100%;
    border-collapse: collapse;
  }}

  thead tr {{
    border-bottom: 1px solid var(--border);
  }}

  th {{
    padding: 10px 16px;
    text-align: left;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--muted);
  }}

  td {{
    padding: 12px 16px;
    border-bottom: 1px solid var(--border);
    vertical-align: middle;
  }}

  tr:hover td {{ background: var(--surface); }}

  .td-num {{ color: var(--muted); width: 36px; font-size: 11px; }}

  .td-title {{
    font-size: 13px;
    font-weight: 500;
    max-width: 260px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }}

  .td-bpm {{
    width: 100px;
    font-size: 22px;
    font-weight: 700;
    color: var(--accent);
    letter-spacing: -0.5px;
  }}
  .td-bpm .unit {{
    font-size: 10px;
    font-weight: 400;
    color: var(--muted);
    letter-spacing: 0;
    margin-left: 2px;
  }}
  .td-bpm.empty {{ color: var(--muted); font-size: 13px; }}

  .td-key {{ width: 130px; }}
  .camelot-badge {{
    display: inline-block;
    background: var(--accent-dim);
    color: var(--accent);
    border: 1px solid rgba(232,255,71,0.3);
    padding: 2px 8px;
    border-radius: 3px;
    font-size: 11px;
    font-weight: 700;
    margin-right: 6px;
    letter-spacing: 0.5px;
  }}
  .key-name {{ font-size: 11px; color: var(--muted); }}

  .td-meta {{ width: 200px; }}

  .bar-wrap {{
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 4px;
  }}
  .bar-label {{
    font-size: 9px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    width: 20px;
    flex-shrink: 0;
  }}
  .bar-bg {{
    flex: 1;
    height: 3px;
    background: var(--border-light);
    border-radius: 2px;
    overflow: hidden;
  }}
  .bar-fill {{
    height: 100%;
    border-radius: 2px;
    transition: width 0.6s ease;
  }}
  .bar-energy {{ background: linear-gradient(90deg, #7B2FFF, #FF4D4D); }}
  .bar-dance {{ background: linear-gradient(90deg, #FF8C00, #E8FF47); }}

  .genre-tag {{
    display: inline-block;
    background: var(--border);
    color: #aaa;
    padding: 2px 8px;
    border-radius: 3px;
    font-size: 10px;
    font-weight: 500;
    margin-top: 4px;
    letter-spacing: 0.5px;
  }}
  .discogs-tag {{
    font-size: 10px;
    color: var(--muted);
    margin-left: 4px;
  }}

  .td-action {{ width: 110px; }}

  .btn-analyze {{
    background: transparent;
    color: var(--text);
    border: 1px solid var(--border-light);
    padding: 6px 14px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s;
    width: 90px;
    text-align: center;
  }}
  .btn-analyze:hover {{ border-color: var(--accent); color: var(--accent); }}
  .btn-analyze:disabled {{ opacity: 0.3; cursor: default; border-color: var(--border); color: var(--muted); }}

  .status-tag {{
    font-size: 10px;
    padding: 4px 10px;
    border-radius: 3px;
    font-weight: 600;
    letter-spacing: 0.5px;
    display: inline-block;
  }}
  .status-done {{ background: rgba(71,255,138,0.1); color: var(--green); }}
  .status-locked {{ background: var(--border); color: var(--muted); }}
  .status-error {{ background: rgba(255,77,77,0.1); color: var(--red); }}
  .status-queue {{ background: var(--border); color: var(--muted); }}

  .loading-dots::after {{
    content: '';
    animation: dots 1.2s steps(3, end) infinite;
  }}
  @keyframes dots {{
    0%   {{ content: '.'; }}
    33%  {{ content: '..'; }}
    66%  {{ content: '...'; }}
    100% {{ content: ''; }}
  }}

  footer {{
    padding: 24px 28px;
    border-top: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    color: var(--muted);
    font-size: 11px;
    margin-top: 8px;
  }}
  footer a {{
    color: var(--accent);
    text-decoration: none;
    font-weight: 600;
  }}
</style>
</head>
<body>

<header>
  <div class="logo">BANDPULSE</div>
  <div class="artist-name">{artist}</div>
  <button class="analyze-all" id="analyzeAllBtn" onclick="analyzeAll()">▶ Analyze All</button>
</header>

<table>
  <thead>
    <tr>
      <th>#</th>
      <th>Track</th>
      <th>BPM</th>
      <th>Key</th>
      <th>Energy / Dance</th>
      <th></th>
    </tr>
  </thead>
  <tbody id="trackList"></tbody>
</table>

<footer>
  <span>BANDPULSE — local analysis, no data leaves your Mac</span>
  <a href="https://bandpulse.app" target="_blank">bandpulse.app</a>
</footer>

<script>
const tracks = {tracks_json};
const queue = [];
let running = false;

function renderTracks() {{
  const tbody = document.getElementById('trackList');
  tbody.innerHTML = '';
  tracks.forEach((t, i) => {{
    const row = document.createElement('tr');
    row.id = 'row' + i;

    let actionCell = '';
    if (t.bpm) {{
      actionCell = '<span class="status-tag status-done">Bandcamp</span>';
    }} else if (t.url) {{
      actionCell = '<button class="btn-analyze" id="btn'+i+'" onclick="enqueue('+i+')">Analyze</button>';
    }} else {{
      actionCell = '<span class="status-tag status-locked">No stream</span>';
    }}

    let bpmCell = t.bpm
      ? '<span style="font-size:22px;font-weight:700;color:var(--accent)">'+t.bpm+'<span class="unit">BPM</span></span>'
      : '<span class="td-bpm empty" id="bpm'+i+'">—</span>';

    row.innerHTML = `
      <td class="td-num">${{i+1}}</td>
      <td class="td-title">${{t.title}}</td>
      <td class="td-bpm" id="bpmcell${{i}}">${{bpmCell}}</td>
      <td class="td-key" id="keycell${{i}}">—</td>
      <td class="td-meta" id="metacell${{i}}">—</td>
      <td class="td-action" id="action${{i}}">${{actionCell}}</td>
    `;
    tbody.appendChild(row);
  }});
}}

function showResult(i, d) {{
  document.getElementById('bpmcell'+i).innerHTML =
    '<span style="font-size:22px;font-weight:700;color:var(--accent)">'+d.bpm+'<span class="unit"> BPM</span></span>';

  const keyHtml = d.camelot !== '?'
    ? '<span class="camelot-badge">'+d.camelot+'</span><span class="key-name">'+d.key+'</span>'
    : '<span class="key-name">'+d.key+'</span>';
  document.getElementById('keycell'+i).innerHTML = keyHtml;

  let metaHtml = '';
  if (d.energy != null) {{
    metaHtml += `<div class="bar-wrap">
      <span class="bar-label">NRG</span>
      <div class="bar-bg"><div class="bar-fill bar-energy" style="width:${{d.energy}}%"></div></div>
      <span style="font-size:10px;color:var(--muted);width:28px">${{d.energy}}%</span>
    </div>`;
  }}
  if (d.danceability != null) {{
    metaHtml += `<div class="bar-wrap">
      <span class="bar-label">DNC</span>
      <div class="bar-bg"><div class="bar-fill bar-dance" style="width:${{d.danceability}}%"></div></div>
      <span style="font-size:10px;color:var(--muted);width:28px">${{d.danceability}}%</span>
    </div>`;
  }}
  if (d.genre) {{
    metaHtml += '<span class="genre-tag">'+d.genre+'</span>';
    if (d.genre_discogs && d.genre_discogs !== d.genre) {{
      metaHtml += '<span class="discogs-tag">'+d.genre_discogs+'</span>';
    }}
  }}
  document.getElementById('metacell'+i).innerHTML = metaHtml || '—';
  document.getElementById('action'+i).innerHTML = '<span class="status-tag status-done">✓ Done</span>';
}}

function showLoading(i) {{
  document.getElementById('bpmcell'+i).innerHTML = '<span class="loading-dots" style="color:var(--muted);font-size:13px">Analyzing</span>';
  const btn = document.getElementById('btn'+i);
  if (btn) {{ btn.disabled = true; btn.textContent = '...'; }}
}}

function enqueue(i) {{
  const t = tracks[i];
  if (!t.url) return;
  const btn = document.getElementById('btn'+i);
  if (btn) {{ btn.disabled = true; btn.textContent = 'Queued'; }}
  queue.push(i);
  if (!running) runQueue();
}}

function runQueue() {{
  if (queue.length === 0) {{ running = false; return; }}
  running = true;
  const i = queue.shift();
  showLoading(i);
  fetch('/analyze?url=' + encodeURIComponent(tracks[i].url))
    .then(r => r.json())
    .then(d => {{
      if (d.error) throw new Error(d.error);
      showResult(i, d);
    }})
    .catch(e => {{
      document.getElementById('action'+i).innerHTML =
        '<span class="status-tag status-error">Error</span>';
      document.getElementById('bpmcell'+i).innerHTML = '—';
    }})
    .finally(() => runQueue());
}}

function analyzeAll() {{
  const allBtn = document.getElementById('analyzeAllBtn');
  allBtn.disabled = true;
  tracks.forEach((t, i) => {{
    if (t.url && !t.bpm) enqueue(i);
  }});
}}

renderTracks();
</script>
</body>
</html>"""
    return Response(html, mimetype="text/html")

if __name__ == "__main__":
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  BANDPULSE Server v1.0")
    print("  http://localhost:5555")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    app.run(host="127.0.0.1", port=5555, debug=False, threaded=True)
