#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="./datasets/video/custom_videos"
OUTPUT_DIR="./datasets/video/custom_videos_standardized"

mkdir -p "$OUTPUT_DIR"

find "$INPUT_DIR" -maxdepth 1 -type f \( \
    -iname "*.mp4" -o \
    -iname "*.mov" -o \
    -iname "*.avi" -o \
    -iname "*.mkv" \) -print0 |
while IFS= read -r -d '' in_file
do
    base="$(basename "$in_file")"
    stem="${base%.*}"
    out_file="$OUTPUT_DIR/${stem}.mp4"

    echo "Processing: $in_file"
    echo "Output: $out_file"

    # -ss 2 \
    ffmpeg -nostdin -y \
        -i "$in_file" \
        -vf "fps=30" \
        -t 60 \
        -c:v libx264 -preset medium -crf 18 \
        -c:a aac -b:a 192k \
        -map 0:v:0 -map 0:a? \
        "$out_file"
done

echo "Done."