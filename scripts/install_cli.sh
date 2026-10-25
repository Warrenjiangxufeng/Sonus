#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
chmod +x "$ROOT/bin/sunus"
if [ -w /usr/local/bin ]; then
  ln -sfn "$ROOT/bin/sunus" /usr/local/bin/sunus
else
  sudo ln -sfn "$ROOT/bin/sunus" /usr/local/bin/sunus
fi
printf 'installed: /usr/local/bin/sunus -> %s/bin/sunus\n' "$ROOT"

