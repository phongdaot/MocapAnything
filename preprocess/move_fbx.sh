#!/bin/bash
# move_fbx.sh - flatten fbx files into their character folder

for char_dir in */; do
    [ -d "$char_dir" ] || continue
    echo "Processing: $char_dir"

    find "$char_dir" -mindepth 2 -type f -iname "*.fbx" | while read -r fbx; do
        filename=$(basename "$fbx")
        target="$char_dir$filename"

        # handle name conflicts by appending a counter
        if [ -e "$target" ]; then
            base="${filename%.*}"
            ext="${filename##*.}"
            i=1
            while [ -e "$char_dir${base}_$i.$ext" ]; do
                i=$((i+1))
            done
            target="$char_dir${base}_$i.$ext"
        fi

        mv "$fbx" "$target"
        echo "  moved: $fbx -> $target"
    done

    # remove empty subdirectories left behind
    find "$char_dir" -mindepth 1 -type d -empty -delete
done

echo "Done."
