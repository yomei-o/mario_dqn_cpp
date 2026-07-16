#!/bin/bash
# Regenerate the README demo GIF from a recorded play (web/run.bin by default).
#
# The training auto-record keeps web/run.bin pointing at the latest best / clear
# (see RESUME.md). To refresh the GIF shown on GitHub:
#
#   tools/make_gif.sh                 # web/run.bin -> docs/mario_1-1_clear.gif
#   tools/make_gif.sh web/run.bin docs/mario_1-1_clear.gif
#   FFMPEG=/path/to/ffmpeg tools/make_gif.sh
#
# ...then `git add docs/mario_1-1_clear.gif && git commit && git push`.
#
# run.bin format (MRUN): "MRUN" + u32 nframes + u32 w + u32 h + nframes*w*h*3 RGB.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
FF="${FFMPEG:-/c/prog/ffmpeg/bin/ffmpeg.exe}"
RUN="${1:-web/run.bin}"
OUT="${2:-docs/mario_1-1_clear.gif}"
[ -f "$RUN" ] || { echo "no recording: $RUN (run: mario_dqn record/recorddemo ...)"; exit 1; }
mkdir -p "$(dirname "$OUT")"
W=$(python -c "import struct,sys;print(struct.unpack('<I',open('$RUN','rb').read(16)[8:12])[0])")
H=$(python -c "import struct,sys;print(struct.unpack('<I',open('$RUN','rb').read(16)[12:16])[0])")
tail -c +17 "$RUN" > /tmp/mrun_raw.rgb          # strip the 16-byte MRUN header
# 2-pass palette GIF (crisp NES colors), ~2x speed (input 30fps -> output 15fps).
"$FF" -y -f rawvideo -pix_fmt rgb24 -s ${W}x${H} -r 30 -i /tmp/mrun_raw.rgb \
      -vf "fps=15,palettegen=stats_mode=diff" /tmp/mrun_pal.png
"$FF" -y -f rawvideo -pix_fmt rgb24 -s ${W}x${H} -r 30 -i /tmp/mrun_raw.rgb -i /tmp/mrun_pal.png \
      -lavfi "fps=15[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3" "$OUT"
echo "wrote $OUT ($(du -h "$OUT" | cut -f1), ${W}x${H})"
