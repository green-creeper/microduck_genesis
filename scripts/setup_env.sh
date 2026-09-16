#!/bin/sh
# Setup virtual environment and ensure compatibility on macOS
set -e

DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$DIR"

echo "== Syncing dependencies with uv..."
uv sync "$@"

# On macOS, ensure .venv/bin/mjpython forwards to python
# (works around MuJoCo wheel's missing @rpath/libpython on standalone uv python)
if [ "$(uname -s)" = "Darwin" ] && [ -f ".venv/bin/mjpython" ]; then
    echo "== Setting up macOS interpreter trampoline..."
    cat << 'EOF' > .venv/bin/mjpython
#!/bin/sh
exec "$(dirname "$0")/python" "$@"
EOF
    chmod +x .venv/bin/mjpython
fi

echo "== Setup complete!"
