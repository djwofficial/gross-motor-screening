#!/bin/bash
# Mac: double-click in Finder to open the live window (webcam + stick figure + live score).
# Auto-picks the first non-black camera via app.main.pick_camera (handles Continuity Camera at index 0).
# The first time, macOS asks to let Terminal use the camera: allow it, then double-click again.
cd "$(dirname "$0")" || exit 1
if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python3; fi
CAM_ARGS=()
if [[ " $* " != *" --camera "* ]]; then
  CAM=$("$PY" -c "from app.main import pick_camera; print(pick_camera())")
  echo "Using camera index: $CAM"
  CAM_ARGS=(--camera "$CAM")
fi
"$PY" -m app.ui "${CAM_ARGS[@]}" --device cpu --pose-mode lightweight --show-score "$@"
code=$?
if [ $code -ne 0 ]; then
  echo
  echo "The window stopped with an error (code $code). The lines above say why."
  read -r -n 1 -s -p "Press any key to close."
fi
exit $code
