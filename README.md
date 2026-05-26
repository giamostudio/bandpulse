# BANDPULSE — BPM & Mood Analyzer for Bandcamp

Analyze BPM, key, energy, danceability and genre of any Bandcamp track.  
Runs entirely on your Mac. No data leaves your machine. Free and open source.

## Features

- **BPM** — accurate tempo detection via Essentia multifeature algorithm
- **Key + Camelot** — harmonic mixing made easy
- **Energy** — track intensity from 0–100%
- **Danceability** — ML-powered crowd-move score
- **Genre** — 400 Discogs categories (Deep House, Dub Techno, Funk, Ambient...)
- **Full album view** — analyze every track in one panel, one by one or all at once

## Requirements

- macOS (Apple Silicon recommended)
- Python 3.10+ with a virtual environment
- Essentia + Essentia-TensorFlow

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/bandpulse.git
cd bandpulse

# 2. Create virtual environment
python3 -m venv env
source env/bin/activate

# 3. Install dependencies
pip install -r server/requirements.txt

# 4. Download ML models
# Place in ~/lupin_models/:
# - discogs-effnet-bs64-1.pb + .json
# - danceability-musicnn-msd-1.pb
# Models available at: https://essentia.upf.edu/models/

# 5. Start the server
python server/bandpulse_server.py
```

## Install the bookmarklet

1. Open `docs/index.html` in Safari
2. Show bookmarks bar: **⌘ Shift B**
3. Drag the **♫ BANDPULSE** button into the bar
4. Go to any Bandcamp page and click it

## How it works

1. Bookmarklet reads track URLs from Bandcamp's `data-tralbum` JSON
2. Opens a local panel at `http://localhost:5555/album`
3. Server downloads the MP3 preview, analyzes with Essentia ML models
4. Returns BPM, key, energy, danceability and genre
5. Temp file is deleted immediately after analysis

## License

MIT — use freely, attribution appreciated.
