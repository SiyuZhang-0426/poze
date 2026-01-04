#!/usr/bin/env python3
"""
Description: Quick viewer for Poze-generated .ply point clouds or sequences.

This helper loads one or more PLY files (a single path, glob, or directory),
optionally downsamples the points, and writes a lightweight mp4/gif preview
with consistent axis limits so you can scrub through Poze outputs quickly.

Example:
    python visualize_ply_sequence.py --input outputs/pi3_points --save outputs/pi3_points_preview.mp4 --max-points 60000 --fps 6

Requires matplotlib (install with: pip install matplotlib).
Also needs plyfile and imageio[ffmpeg] (pip install plyfile imageio[ffmpeg]).
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from plyfile import PlyData

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import imageio  # noqa: E402


PointCloud = Tuple[np.ndarray, np.ndarray | None, str]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize Poze-generated PLY point clouds or sequences."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a .ply file, a directory of PLYs, or a glob pattern like 'outputs/pi3_points/*.ply'.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=Path("outputs/ply_preview.mp4"),
        help="Output video or gif path (extension determines format).",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=80000,
        help="Randomly subsample to this many points per frame to keep rendering light.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Use every Nth frame from the sequence (e.g., 2 keeps every other ply).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=6,
        help="Preview frame rate for the saved animation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for downsampling (omit to vary samples each run).",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=0.5,
        help="Matplotlib scatter point size.",
    )
    parser.add_argument(
        "--title",
        default="Poze point cloud",
        help="Title to overlay on the preview.",
    )
    parser.add_argument(
        "--elev",
        type=float,
        default=20.0,
        help="Elevation angle for the 3D view (degrees).",
    )
    parser.add_argument(
        "--azim",
        type=float,
        default=-60.0,
        help="Azimuth angle for the 3D view (degrees).",
    )
    parser.add_argument(
        "--pad-fraction",
        type=float,
        default=0.05,
        help="Axis padding as a fraction of the max span.",
    )
    parser.add_argument(
        "--min-pad",
        type=float,
        default=0.1,
        help="Minimum absolute axis padding.",
    )
    return parser.parse_args()


def _expand_sources(source: str, stride: int) -> List[Path]:
    pattern_chars = ("*", "?", "[")
    if any(char in source for char in pattern_chars):
        paths = [Path(p) for p in glob.glob(source)]
    else:
        src_path = Path(source)
        if src_path.is_dir():
            paths = list(src_path.glob("*.ply"))
        else:
            paths = [src_path]

    stride = max(stride, 1)
    paths = sorted(paths)[::stride]
    if not paths:
        raise FileNotFoundError(f"No PLY files found for {source}")
    return paths


def _load_point_cloud(path: Path) -> Tuple[np.ndarray, np.ndarray | None]:
    ply_data = PlyData.read(str(path))
    try:
        vertex = ply_data["vertex"].data
    except KeyError as exc:
        raise ValueError(f"{path} does not contain a 'vertex' element") from exc

    if len(vertex) == 0:
        raise ValueError(f"{path} contains no vertex data")
    required_fields = ("x", "y", "z")
    missing_fields = [f for f in required_fields if f not in vertex.dtype.names]
    if missing_fields:
        raise ValueError(f"{path} is missing vertex fields: {', '.join(missing_fields)}")

    xyz = np.column_stack([vertex[f] for f in required_fields]).astype(np.float32)

    has_color = {"red", "green", "blue"}.issubset(vertex.dtype.names)
    rgb = None
    if has_color:
        rgb = (
            np.vstack((vertex["red"], vertex["green"], vertex["blue"]))
            .T.astype(np.float32)
            / 255.0
        )
    return xyz, rgb


def _maybe_downsample(
    xyz: np.ndarray, rgb: np.ndarray | None, max_points: int, seed: int | None
) -> Tuple[np.ndarray, np.ndarray | None]:
    if max_points < 0:
        raise ValueError("max_points must be non-negative")
    if max_points == 0 or len(xyz) <= max_points:
        return xyz, rgb

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(xyz), size=max_points, replace=False)
    xyz_ds = xyz[idx]
    rgb_ds = rgb[idx] if rgb is not None else None
    return xyz_ds, rgb_ds


def _prepare_frames(
    paths: Sequence[Path], max_points: int, seed: int | None
) -> List[PointCloud]:
    frames: List[PointCloud] = []
    for p in paths:
        xyz, rgb = _load_point_cloud(p)
        xyz, rgb = _maybe_downsample(xyz, rgb, max_points, seed)
        frames.append((xyz, rgb, p.name))
    return frames


def _global_bounds(
    frames: Sequence[PointCloud], pad_fraction: float, min_pad: float
) -> Tuple[np.ndarray, np.ndarray]:
    all_points = np.concatenate([xyz for xyz, _, _ in frames], axis=0)
    low = all_points.min(axis=0)
    high = all_points.max(axis=0)
    span = (high - low).max()
    pad = max(min_pad, pad_fraction * span) if span > 0 else min_pad
    return low - pad, high + pad


def _render_sequence(
    frames: Sequence[PointCloud],
    save_path: Path,
    fps: int,
    title: str,
    elev: float,
    azim: float,
    point_size: float,
    pad_fraction: float,
    min_pad: float,
) -> None:
    if not frames:
        raise ValueError("No valid PLY frames to render.")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    low, high = _global_bounds(frames, pad_fraction, min_pad)
    ax.set_xlim(low[0], high[0])
    ax.set_ylim(low[1], high[1])
    ax.set_zlim(low[2], high[2])
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    scatter = None

    with imageio.get_writer(save_path, fps=fps) as writer:
        # Frames are preloaded; stream them to the writer to keep the encoder footprint small.
        for idx, (xyz, rgb, label) in enumerate(frames, start=1):
            if scatter is not None:
                scatter.remove()

            if rgb is not None:
                colors = rgb
            else:
                z_values = xyz[:, 2]
                z_min = z_values.min()
                z_range = z_values.max() - z_min
                if z_range <= 0:
                    colors = plt.cm.viridis(0.5)
                else:
                    z_norm = (z_values - z_min) / z_range
                    colors = plt.cm.viridis(z_norm)

            scatter = ax.scatter(
                xyz[:, 0],
                xyz[:, 1],
                xyz[:, 2],
                s=point_size,
                depthshade=False,
                c=colors,
            )

            ax.set_title(f"{title} — frame {idx}/{len(frames)} ({label})")
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            writer.append_data(image)

    plt.close(fig)


def main() -> None:
    args = _parse_args()
    ply_paths = _expand_sources(args.input, args.stride)
    frames = _prepare_frames(ply_paths, args.max_points, args.seed)
    _render_sequence(
        frames,
        args.save,
        args.fps,
        args.title,
        args.elev,
        args.azim,
        args.point_size,
        args.pad_fraction,
        args.min_pad,
    )
    print(f"Saved preview to {args.save} ({len(frames)} frame(s)).")


if __name__ == "__main__":
    main()
