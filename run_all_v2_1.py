"""Run pipeline v2.1 on all videos sequentially in separate processes."""
import os
import sys
import time
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(BASE_DIR, "outputs_v2_1")

VIDEOS = [
    "Video_evaluation_01.mp4",  # 81s
    "CTML_03_01.mp4",           # 136s
    "A0.mp4",                    # 217s
    "A1.mp4",                    # 217s
    "A3.mp4",                    # 217s
    "A5.mp4",                    # 217s
    "test_video.mp4",            # 477s
]

CMD_TEMPLATE = [
    sys.executable, "-m", "pipeline_v2_1.main",
    "--output-root", OUTPUT_ROOT,
    "--no-grounding-dino",
    "--skip-vlm",
    "--no-siglip",
    "--importance-backend", "heuristic",
    "--no-ocr",
    "--video",
]


def main():
    total_start = time.time()
    results = {}

    for i, video_name in enumerate(VIDEOS, 1):
        video_path = os.path.join(BASE_DIR, video_name)
        if not os.path.exists(video_path):
            print(f"[{i}/{len(VIDEOS)}] SKIP {video_name} (not found)")
            continue

        print(f"\n{'='*60}")
        print(f"[{i}/{len(VIDEOS)}] Processing: {video_name}")
        print(f"{'='*60}")

        cmd = CMD_TEMPLATE + [video_path]
        t0 = time.time()

        try:
            proc = subprocess.run(
                cmd, cwd=BASE_DIR, timeout=600,
                capture_output=False,  # let output stream to console
            )
            elapsed = time.time() - t0
            if proc.returncode == 0:
                print(f"[{i}/{len(VIDEOS)}] OK: {video_name} ({elapsed:.0f}s)")
                results[video_name] = ("OK", elapsed)
            else:
                print(f"[{i}/{len(VIDEOS)}] FAIL: {video_name} (exit {proc.returncode}, {elapsed:.0f}s)")
                results[video_name] = ("FAIL", elapsed)
        except subprocess.TimeoutExpired:
            elapsed = time.time() - t0
            print(f"[{i}/{len(VIDEOS)}] TIMEOUT: {video_name} ({elapsed:.0f}s)")
            results[video_name] = ("TIMEOUT", elapsed)
        except Exception as e:
            elapsed = time.time() - t0
            print(f"[{i}/{len(VIDEOS)}] ERROR: {video_name}: {e}")
            results[video_name] = ("ERROR", elapsed)

    total = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"ALL DONE. Total: {total/60:.1f} minutes")
    print(f"{'='*60}")
    for name, (status, elapsed) in results.items():
        print(f"  {name}: {status} ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
