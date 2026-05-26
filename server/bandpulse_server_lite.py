#!/usr/bin/env python3
"""
BANDPULSE Server Lite — BPM & Key analyzer for Bandcamp
No ML models needed. Uses librosa.
"""
from flask import Flask, request, jsonify, Response
import tempfile, os, json, base64, threading
import requests as req_lib

app = Flask(__name__)
_lock = threading.Lock()

CAMELOT = {
    "B major": "1B",  "G# minor": "1A",
    "F# major": "2B", "D# minor": "2A", "Eb minor": "2A",
    "C# major": "3B", "Db major": "3B", "A# minor": "3A", "Bb minor": "3A",
    "Ab major": "4B", "G# major": "4B", "F minor": "4A",
    "Eb major": "5B", "D# major": "5B", "C minor": "5A",
    "Bb major": "6B", "A# major": "6B", "G minor": "6A",
    "F major":  "7B",                   "D minor": "7A",
    "C major":  "8B",                   "A minor": "8A",
    "G major":  "9B",                   "E minor": "9A",
    "D major": "10B",                   "B minor": "10A",
    "A major": "11B",                   "F# minor": "11A",
    "E major": "12B",                   "C# minor": "12A", "Db minor": "12A",
}

PITCH_CLASSES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
MAJOR_PROFILE = [6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88]
MINOR_PROFILE = [6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17]

def detect_key(y, sr):
    import numpy as np
    import librosa
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    best_score, best_key = -1, "C major"
    for i in range(12):
        rotated = np.roll(chroma_mean, -i)
        maj = np.corrcoef(rotated, MAJOR_PROFILE)[0,1]
        mn  = np.corrcoef(rotated, MINOR_PROFILE)[0,1]
        if maj > best_score:
            best_score, best_key = maj, f"{PITCH_CLASSES[i]} major"
        if mn > best_score:
            best_score, best_key = mn, f"{PITCH_CLASSES[i]} minor"
    return best_key

def analyze_audio(path):
    import librosa
    import numpy as np
    y, sr = librosa.load(path, sr=None, mono=True)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = round(float(np.atleast_1d(tempo)[0]), 1)
    key = detect_key(y, sr)
    rms = float(np.sqrt(np.mean(y**2)))
    energy = min(100, round(rms * 3000))
    return bpm, key, energy

def to_camelot(k):
    return CAMELOT.get(k.strip(), "?")

@app.after_request
def cors(r):
    r.headers["Access-Control-Allow-Origin"] = "*"
    return r

@app.route("/ping")
def ping():
    return jsonify({"ok": True, "version": "1.0-lite"})

@app.route("/analyze")
def analyze():
    url = request.args.get("url")
    if not url:
        return jsonify({"error": "missing url"}), 400
    tmp_path = None
    try:
        with _lock:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp_path = tmp.name
            r = req_lib.get(url, timeout=30, stream=True,
                            headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(65536):
                    f.write(chunk)
            bpm, key, energy = analyze_audio(tmp_path)
            os.unlink(tmp_path)
        return jsonify({
            "bpm": bpm,
            "key": key,
            "camelot": to_camelot(key),
            "energy": energy,
            "danceability": None,
            "genre": None,
            "genre_discogs": None,
        })
    except Exception as e:
        if tmp_path:
            try: os.unlink(tmp_path)
            except: pass
        return jsonify({"error": str(e)}), 500

@app.route("/album")
def album():
    raw = request.args.get("data", "")
    artist = request.args.get("artist", "Album")
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
<title>BANDPULSE — {artist}</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700;900&display=swap" rel="stylesheet">
<style>
  :root{{--bg:#0A0A0A;--surface:#111;--border:#1E1E1E;--border2:#2A2A2A;--text:#F0EDE8;--muted:#555;--accent:#E8FF47;--green:#47FF8A;--red:#FF4D4D;}}
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:var(--bg);color:var(--text);font-family:'Inter',sans-serif;font-size:13px;min-height:100vh}}
  header{{display:flex;align-items:center;justify-content:space-between;padding:18px 24px;border-bottom:1px solid var(--border)}}
  .logo{{font-size:12px;font-weight:900;letter-spacing:3px;color:var(--accent)}}
  .artist{{font-size:12px;color:var(--muted);max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
  .all-btn{{background:var(--accent);color:#000;border:none;padding:7px 18px;border-radius:4px;font-size:11px;font-weight:700;letter-spacing:1px;text-transform:uppercase;cursor:pointer}}
  .all-btn:disabled{{opacity:.3;cursor:default}}
  table{{width:100%;border-collapse:collapse}}
  th{{padding:8px 16px;text-align:left;font-size:9px;font-weight:600;letter-spacing:1.5px;text-transform:uppercase;color:var(--muted);border-bottom:1px solid var(--border)}}
  td{{padding:11px 16px;border-bottom:1px solid var(--border);vertical-align:middle}}
  tr:hover td{{background:var(--surface)}}
  .n{{color:var(--muted);width:32px;font-size:11px}}
  .t{{font-size:13px;font-weight:500;max-width:280px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
  .b{{width:96px;font-size:22px;font-weight:900;color:var(--accent);letter-spacing:-0.5px}}
  .bu{{font-size:9px;font-weight:400;color:var(--muted)}}
  .b.empty{{color:var(--muted);font-size:13px}}
  .k{{width:130px}}
  .cam{{display:inline-block;background:rgba(232,255,71,.1);color:var(--accent);border:1px solid rgba(232,255,71,.25);padding:1px 7px;border-radius:3px;font-size:10px;font-weight:700;margin-right:5px}}
  .kn{{font-size:11px;color:var(--muted)}}
  .m{{width:160px}}
  .bw{{display:flex;align-items:center;gap:5px;margin-bottom:3px}}
  .bl{{font-size:8px;color:var(--muted);text-transform:uppercase;letter-spacing:1px;width:20px}}
  .bg{{flex:1;height:3px;background:#222;border-radius:2px;overflow:hidden}}
  .bf{{height:100%;border-radius:2px}}
  .be{{background:linear-gradient(90deg,#7B2FFF,#FF4D4D)}}
  .ac{{width:100px}}
  .btn{{background:transparent;color:var(--text);border:1px solid var(--border2);padding:5px 14px;border-radius:4px;font-size:11px;font-weight:500;cursor:pointer;width:88px;text-align:center}}
  .btn:hover{{border-color:var(--accent);color:var(--accent)}}
  .btn:disabled{{opacity:.3;cursor:default;border-color:var(--border);color:var(--muted)}}
  .tag{{font-size:10px;padding:3px 10px;border-radius:3px;font-weight:600;letter-spacing:.5px;display:inline-block}}
  .done{{background:rgba(71,255,138,.1);color:var(--green)}}
  .locked{{background:var(--border);color:var(--muted)}}
  .err{{background:rgba(255,77,77,.1);color:var(--red)}}
  .ld::after{{content:'';animation:dots 1s steps(3,end) infinite}}
  @keyframes dots{{0%{{content:'.';}}33%{{content:'..';}}66%{{content:'...';}}}}
  footer{{padding:20px 24px;border-top:1px solid var(--border);display:flex;justify-content:space-between;color:var(--muted);font-size:11px;margin-top:8px}}
  footer a{{color:var(--accent);text-decoration:none;font-weight:600}}
</style>
</head>
<body>
<header>
  <div class="logo">BANDPULSE</div>
  <div class="artist">{artist}</div>
  <button class="all-btn" id="allBtn" onclick="analyzeAll()">▶ Analyze All</button>
</header>
<table>
  <thead><tr><th>#</th><th>Track</th><th>BPM</th><th>Key</th><th>Energy</th><th></th></tr></thead>
  <tbody id="tb"></tbody>
</table>
<footer>
  <span>BANDPULSE — local analysis, no data leaves your Mac</span>
  <a href="https://giamostudio.github.io/bandpulse" target="_blank">giamostudio.github.io/bandpulse</a>
</footer>
<script>
const tracks={tracks_json};
const queue=[];let running=false;
function render(){{
  const tb=document.getElementById('tb');
  tb.innerHTML='';
  tracks.forEach((t,i)=>{{
    const row=document.createElement('tr');
    row.id='r'+i;
    let ac=t.bpm
      ?'<span class="tag done">Bandcamp</span>'
      :t.url
        ?'<button class="btn" id="btn'+i+'" onclick="enq('+i+')">Analyze</button>'
        :'<span class="tag locked">No stream</span>';
    let bc=t.bpm
      ?'<span style="font-size:22px;font-weight:900;color:var(--accent)">'+t.bpm+'<span class="bu"> BPM</span></span>'
      :'<span class="b empty" id="bc'+i+'">—</span>';
    row.innerHTML=`<td class="n">${{i+1}}</td><td class="t">${{t.title}}</td><td class="b" id="bc${{i}}">${{bc}}</td><td class="k" id="kc${{i}}">—</td><td class="m" id="mc${{i}}">—</td><td class="ac" id="ac${{i}}">${{ac}}</td>`;
    tb.appendChild(row);
  }});
}}
function showRes(i,d){{
  document.getElementById('bc'+i).innerHTML='<span style="font-size:22px;font-weight:900;color:var(--accent)">'+d.bpm+'<span class="bu"> BPM</span></span>';
  document.getElementById('kc'+i).innerHTML=(d.camelot&&d.camelot!='?'?'<span class="cam">'+d.camelot+'</span>':'')+'<span class="kn">'+d.key+'</span>';
  let m='';
  if(d.energy!=null)m+='<div class="bw"><span class="bl">NRG</span><div class="bg"><div class="bf be" style="width:'+d.energy+'%"></div></div><span style="font-size:10px;color:var(--muted);width:28px">'+d.energy+'%</span></div>';
  document.getElementById('mc'+i).innerHTML=m||'—';
  document.getElementById('ac'+i).innerHTML='<span class="tag done">✓ Done</span>';
}}
function showLoad(i){{
  document.getElementById('bc'+i).innerHTML='<span class="ld" style="color:var(--muted)">Analyzing</span>';
  const b=document.getElementById('btn'+i);if(b){{b.disabled=true;b.textContent='...';}}
}}
function enq(i){{
  if(!tracks[i].url)return;
  const b=document.getElementById('btn'+i);if(b){{b.disabled=true;b.textContent='Queued';}}
  queue.push(i);if(!running)run();
}}
function run(){{
  if(!queue.length){{running=false;return;}}
  running=true;const i=queue.shift();showLoad(i);
  fetch('/analyze?url='+encodeURIComponent(tracks[i].url))
    .then(r=>r.json())
    .then(d=>{{if(d.error)throw new Error(d.error);showRes(i,d);}})
    .catch(()=>{{document.getElementById('ac'+i).innerHTML='<span class="tag err">Error</span>';document.getElementById('bc'+i).innerHTML='—';}})
    .finally(run);
}}
function analyzeAll(){{
  document.getElementById('allBtn').disabled=true;
  tracks.forEach((t,i)=>{{if(t.url&&!t.bpm)enq(i);}});
}}
render();
</script>
</body></html>"""
    return Response(html, mimetype="text/html")

if __name__ == "__main__":
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  BANDPULSE Server v1.0 (lite)")
    print("  http://localhost:5555")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    app.run(host="127.0.0.1", port=5555, debug=False, threaded=True)
