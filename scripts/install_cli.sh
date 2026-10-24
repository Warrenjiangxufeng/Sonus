#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
chmod +x "$ROOT/bin/sonus"
if [ -w /usr/local/bin ]; then
  ln -sfn "$ROOT/bin/sonus" /usr/local/bin/sonus
else
  sudo ln -sfn "$ROOT/bin/sonus" /usr/local/bin/sonus
fi
printf 'installed: /usr/local/bin/sonus -> %s/bin/sonus\n' "$ROOT"

