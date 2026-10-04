"""Generate an embedded-PNG fixture for end-to-end MP4 conversion/playback tests."""

import argparse
import io
from pathlib import Path

import pyarrow as pa
from PIL import Image, ImageDraw

from ._demo import _MARKER, _MARKER_TEXT, _write_json, _write_jsonl, _write_table


def generate_image_demo(output=None):
    root = Path(output or Path(__file__).resolve().parents[1] / "demo_data/demo_z_images")
    if (
        root.is_symlink()
        or root.exists()
        and (
            not (root / _MARKER).is_file()
            or (root / _MARKER).is_symlink()
            or (root / _MARKER).read_text() != _MARKER_TEXT
        )
    ):
        raise ValueError("Refusing to replace an unmarked image demo")
    cameras = ["camera_png", "camera_jpeg"]
    rows = []
    for i in range(240):
        row = {"episode_index": 0, "frame_index": i, "timestamp": i / 30, "index": i}
        for camera in cameras:
            image = Image.new("RGB", (320, 180), (30, 40, 50))
            draw = ImageDraw.Draw(image)
            x = 20 + i % 240
            draw.rectangle((x, 60, x + 40, 110), fill=(230, 170, 50))
            draw.text((15, 15), f"{camera} {i}", fill="white")
            encoded = io.BytesIO()
            image.save(encoded, format="PNG" if camera == cameras[0] else "JPEG")
            row[camera] = {"bytes": encoded.getvalue(), "path": None}
        rows.append(row)
    _write_table(root / "data/chunk-000/episode_000000.parquet", pa.Table.from_pylist(rows))
    _write_json(
        root / "meta/info.json",
        {
            "codebase_version": "v2.1",
            "fps": 30,
            "total_episodes": 1,
            "total_frames": 240,
            "features": {camera: {"dtype": "image", "shape": [180, 320, 3]} for camera in cameras},
        },
    )
    _write_jsonl(root / "meta/episodes.jsonl", [{"episode_index": 0, "length": 240, "tasks": []}])
    (root / _MARKER).write_text(_MARKER_TEXT)
    return root


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    print(generate_image_demo(parser.parse_args().output))
