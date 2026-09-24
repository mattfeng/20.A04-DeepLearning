"""Preview MNIST digits with Kitty graphics or ASCII art (standard library only)."""

import argparse
import base64
import gzip
import os
from pathlib import Path
import secrets
import shutil
import struct
import subprocess
import sys
from urllib.request import urlretrieve


BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
# First 64 entries of Kitty's rowcolumn-diacritics.txt, in protocol order.
# https://sw.kovidgoyal.net/kitty/graphics-protocol/#unicode-placeholders
DIACRITICS = tuple(chr(int(code, 16)) for code in """
0305 030D 030E 0310 0312 033D 033E 033F 0346 034A 034B 034C 0350 0351 0352 0357
035B 0363 0364 0365 0366 0367 0368 0369 036A 036B 036C 036D 036E 036F 0483 0484
0485 0486 0487 0592 0593 0594 0595 0597 0598 0599 059C 059D 059E 059F 05A0 05A1
05A8 05A9 05AB 05AC 05AF 05C4 0610 0611 0612 0613 0614 0615 0616 0617 0657 0658
""".split())


def choose_renderer(requested):
    if requested != "auto":
        return requested
    # Conservative detection: unknown terminals and redirected output use ASCII.
    if not sys.stdout.isatty() or not (
        os.environ.get("KITTY_WINDOW_ID") or os.environ.get("TERM") == "xterm-kitty"
    ):
        return "ascii"
    if os.environ.get("TMUX"):
        if (sys.stdout.encoding or "").lower().replace("-", "") != "utf8":
            return "ascii"
        try:
            result = subprocess.run(
                ["tmux", "show-options", "-p", "-v", "allow-passthrough"],
                capture_output=True, text=True, timeout=2, check=True,
            )
        except (OSError, subprocess.SubprocessError):
            return "ascii"
        if result.stdout.strip() not in ("on", "all"):
            return "ascii"
    return "kitty"


def display_ascii(pixels, rows, cols):
    shades = " .:-=+*#%@"
    for row in range(rows):
        line = pixels[row * cols:(row + 1) * cols]
        # Double characters compensate for tall terminal cells.
        print("".join(shades[pixel * (len(shades) - 1) // 255] * 2 for pixel in line))


def display_tmux_image(payload, rows, cols, height):
    # Reserve real text cells so tmux can move/clip the image with its pane.
    width = min(height * 2, len(DIACRITICS), shutil.get_terminal_size().columns)
    image_id = secrets.randbelow(0xFFFFFF) + 1
    command = (
        f"\033_Ga=T,f=24,t=d,s={cols},v={rows},r={height},c={width},"
        f"i={image_id},U=1,q=2;{payload}\033\\"
    )
    # Only graphics commands bypass tmux; placeholder text goes through tmux.
    print("\033Ptmux;" + command.replace("\033", "\033\033") + "\033\\", end="")
    color = f"\033[38;2;{image_id >> 16};{(image_id >> 8) & 255};{image_id & 255}m"
    for row in range(height):
        cells = "".join(
            "\U0010eeee" + DIACRITICS[row] + DIACRITICS[column]
            for column in range(width)
        )
        print(color + cells + "\033[39m", end="\r\n")
    print(flush=True)


def display_image(pixels, rows, cols, height):
    # Kitty accepts RGB pixels; repeat each grayscale value across R, G, B.
    rgb = bytes(channel for pixel in pixels for channel in (pixel, pixel, pixel))
    payload = base64.b64encode(rgb).decode("ascii")
    if os.environ.get("TMUX"):
        display_tmux_image(payload, rows, cols, height)
        return
    # A 28x28 RGB image fits in one payload (3136 bytes, below the 4096 limit).
    # Set only the display height to preserve the image's aspect ratio.
    print(
        f"\033_Ga=T,f=24,t=d,s={cols},v={rows},r={height},q=2;"
        f"{payload}\033\\",
        end="\r\n", flush=True,
    )


def read_file(data_dir, filename):
    path = data_dir / filename
    if not path.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {filename}...")
        urlretrieve(BASE_URL + filename, path)
    return gzip.decompress(path.read_bytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--index", type=int, default=0, help="first sample (zero-based)")
    parser.add_argument("--count", type=int, default=3, help="number of samples")
    parser.add_argument("--height", type=int, default=10, help="Kitty image height in terminal rows")
    parser.add_argument(
        "--renderer", choices=("auto", "kitty", "ascii"), default="auto",
        help="auto detects Kitty and otherwise uses ASCII (default: auto)",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path(__file__).resolve().parent / "data",
        help="dataset cache directory (default: data next to this script)",
    )
    args = parser.parse_args()
    if args.index < 0 or args.count < 1:
        parser.error("--index must be nonnegative and --count must be positive")
    if args.height < 1:
        parser.error("--height must be positive")
    renderer = choose_renderer(args.renderer)
    if renderer == "kitty" and os.environ.get("TMUX") and args.height > len(DIACRITICS):
        parser.error(f"--height must be at most {len(DIACRITICS)} inside tmux")

    prefix = "train" if args.split == "train" else "t10k"
    images = read_file(args.data_dir, f"{prefix}-images-idx3-ubyte.gz")
    labels = read_file(args.data_dir, f"{prefix}-labels-idx1-ubyte.gz")
    magic, total, rows, cols = struct.unpack_from(">IIII", images)
    label_magic, label_count = struct.unpack_from(">II", labels)
    if (
        magic != 2051 or label_magic != 2049 or total != label_count
        or rows != 28 or cols != 28
        or len(images) != 16 + total * rows * cols
        or len(labels) != 8 + total
    ):
        parser.error("invalid MNIST image or label data")
    if args.index >= total:
        parser.error(f"--index must be less than {total}")

    for index in range(args.index, min(args.index + args.count, total)):
        print(f"\n{args.split} sample {index} | label: {labels[8 + index]}")
        offset = 16 + index * rows * cols
        pixels = images[offset:offset + rows * cols]
        if renderer == "ascii":
            display_ascii(pixels, rows, cols)
        else:
            display_image(pixels, rows, cols, args.height)


if __name__ == "__main__":
    main()
