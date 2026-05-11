# #!/bin/bash

for video in *.mp4; do
  # 去除文件扩展名作为输出文件夹名
  folder_name="${video%.mp4}"
  mkdir -p "$folder_name"
  
  # 使用 ffmpeg 提取帧，保存为 folder_name/xxxx.png
  ffmpeg -i "$video" "$folder_name/%04d.png"
done