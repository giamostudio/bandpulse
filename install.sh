#!/bin/bash
set -e

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  BANDPULSE — Installer"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check Python3
if ! command -v python3 &>/dev/null; then
  echo "❌ Python3 not found. Install it from https://python.org"
  exit 1
fi

INSTALL_DIR="$HOME/bandpulse"
mkdir -p "$INSTALL_DIR"

echo "→ Creating virtual environment..."
python3 -m venv "$INSTALL_DIR/env"

echo "→ Installing dependencies (librosa, flask, requests)..."
"$INSTALL_DIR/env/bin/pip" install --quiet --upgrade pip
"$INSTALL_DIR/env/bin/pip" install --quiet flask requests librosa numpy

echo "→ Downloading server..."
curl -fsSL "https://raw.githubusercontent.com/giamostudio/bandpulse/main/server/bandpulse_server_lite.py" \
  -o "$INSTALL_DIR/bandpulse_server.py"

echo "→ Creating launcher..."
cat > "$INSTALL_DIR/start.command" << 'EOF'
#!/bin/bash
cd "$HOME/bandpulse"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  BANDPULSE Server"
echo "  http://localhost:5555"
echo "  Press Ctrl+C to stop"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
"$HOME/bandpulse/env/bin/python" "$HOME/bandpulse/bandpulse_server.py"
EOF
chmod +x "$INSTALL_DIR/start.command"

echo ""
echo "✅ BANDPULSE installed!"
echo ""
echo "To start the server, double-click:"
echo "  ~/bandpulse/start.command"
echo ""
echo "Or run in Terminal:"
echo "  ~/bandpulse/start.command"
echo ""

# Auto-start now
"$INSTALL_DIR/env/bin/python" "$INSTALL_DIR/bandpulse_server.py"
