#!/usr/bin/env python3
"""Extract frames at regular intervals to map the full video content timeline."""
import os, subprocess

video = "A0.mp4"
out_dir = "outputs/freeze_detection/timeline_A0"
os.makedirs(out_dir, exist_ok=True)

# Extract every 5 seconds for full overview, plus finer around delay points
timestamps = list(range(0, 217, 5))

# Add finer timestamps around delay points and between them
for t in [70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,  # DP1 region
          85, 90, 95, 100, 105, 108, 110, 112, 113, 114, 115, 116, 117, 118, 119, 120,  # between DP1-DP2
          125, 130, 135, 138, 140, 142, 144, 146, 148,  # between DP2-DP3
          149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160,  # DP3 region
          162, 164, 166, 168, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180]:  # DP4 region
    if t not in timestamps:
        timestamps.append(t)

timestamps.sort()

for t in timestamps:
    fname = os.path.join(out_dir, f"t{t:04d}.png")
    if not os.path.exists(fname):
        subprocess.run([
            "ffmpeg", "-ss", str(t), "-i", video,
            "-frames:v", "1", "-y", "-v", "quiet", fname
        ], capture_output=True)

print(f"Extracted {len(timestamps)} frames to {out_dir}/")
