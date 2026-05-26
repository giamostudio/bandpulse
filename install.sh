#!/bin/bash
set -e

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  BANDPULSE — Installer v1.0"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check Python3.11+
PY=""
for cmd in python3.11 python3.12 python3.10 python3; do
  if command -v $cmd &>/dev/null; then
    ver=$($cmd -c "import sys; print(sys.version_info.minor)" 2>/dev/null)
    maj=$($cmd -c "import sys; print(sys.version_info.major)" 2>/dev/null)
    if [ "$maj" = "3" ] && [ "$ver" -ge "9" ] 2>/dev/null; then
      PY=$cmd; break
    fi
  fi
done

if [ -z "$PY" ]; then
  echo "❌ Python 3.9+ not found."
  echo "   Install from: https://www.python.org/downloads/"
  exit 1
fi

echo "→ Using $($PY --version)"

INSTALL_DIR="$HOME/bandpulse"
mkdir -p "$INSTALL_DIR"
mkdir -p "$HOME/bandpulse_models"

echo "→ Creating virtual environment..."
$PY -m venv "$INSTALL_DIR/env"

echo "→ Installing dependencies (this takes ~2 minutes)..."
"$INSTALL_DIR/env/bin/pip" install --quiet --upgrade pip
"$INSTALL_DIR/env/bin/pip" install --quiet essentia-tensorflow flask requests numpy

echo "→ Downloading ML models (19MB)..."
curl -fsSL -o /tmp/bandpulse_models.zip \
  "https://github.com/giamostudio/bandpulse/releases/download/v1.0/bandpulse_models.zip"
unzip -q -o /tmp/bandpulse_models.zip -d "$HOME/bandpulse_models"
rm /tmp/bandpulse_models.zip
echo "   Models OK → ~/bandpulse_models/"

echo "→ Downloading server..."
curl -fsSL "https://raw.githubusercontent.com/giamostudio/bandpulse/main/server/bandpulse_server.py" \
  -o "$INSTALL_DIR/bandpulse_server.py"

echo "→ Creating launcher (start.command)..."
cat > "$INSTALL_DIR/start.command" << 'LAUNCHER'
#!/bin/bash
clear
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  BANDPULSE Server"
echo "  Running at http://localhost:5555"
echo "  Keep this window open."
echo "  Press Ctrl+C to stop."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
"$HOME/bandpulse/env/bin/python" "$HOME/bandpulse/bandpulse_server.py"
LAUNCHER
chmod +x "$INSTALL_DIR/start.command"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✅ BANDPULSE installed!"
echo ""
echo "  Next time, start with:"
echo "  ~/bandpulse/start.command"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Start server now
"$INSTALL_DIR/env/bin/python" "$INSTALL_DIR/bandpulse_server.py"
