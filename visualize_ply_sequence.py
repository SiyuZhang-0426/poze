#!/usr/bin/env python3
"""
Caption: Quick viewer for Poze-generated .ply point clouds or sequences.

This helper loads one or more PLY files (a single path, glob, or directory),
optionally downsamples the points, and writes a lightweight mp4/gif preview
with consistent axis limits so you can scrub through Poze outputs quickly.

Example:
    python visualize_ply_sequence.py \\
        --input outputs/pi3_points \\
        --save outputs/pi3_points_preview.mp4 \\
        --max-points 60000 \\
        --fps 6

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

    paths = sorted(paths)[:: max(stride, 1)]
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
    xyz = np.vstack((vertex["x"], vertex["y"], vertex["z"])).T.astype(np.float32)

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
    xyz: np.ndarray, rgb: np.ndarray | None, max_points: int
) -> Tuple[np.ndarray, np.ndarray | None]:
    if max_points is None or max_points <= 0 or len(xyz) <= max_points:
        return xyz, rgb

    rng = np.random.default_rng(0)
    idx = rng.choice(len(xyz), size=max_points, replace=False)
    xyz_ds = xyz[idx]
    rgb_ds = rgb[idx] if rgb is not None else None
    return xyz_ds, rgb_ds


def _prepare_frames(paths: Sequence[Path], max_points: int) -> List[PointCloud]:
    frames: List[PointCloud] = []
    for p in paths:
        xyz, rgb = _load_point_cloud(p)
        xyz, rgb = _maybe_downsample(xyz, rgb, max_points)
        frames.append((xyz, rgb, p.name))
    return frames


def _global_bounds(frames: Sequence[PointCloud]) -> Tuple[np.ndarray, np.ndarray]:
    mins = []
    maxs = []
    for xyz, _, _ in frames:
        mins.append(np.min(xyz, axis=0))
        maxs.append(np.max(xyz, axis=0))
    low = np.vstack(mins).min(axis=0)
    high = np.vstack(maxs).max(axis=0)
    span = (high - low).max()
    pad = 0.05 * span if span > 0 else 0.1
    return low - pad, high + pad


def _render_sequence(
    frames: Sequence[PointCloud],
    save_path: Path,
    fps: int,
    title: str,
    elev: float,
    azim: float,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    low, high = _global_bounds(frames)
    ax.set_xlim(low[0], high[0])
    ax.set_ylim(low[1], high[1])
    ax.set_zlim(low[2], high[2])
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    scatter = None

    with imageio.get_writer(save_path, fps=fps) as writer:
        for idx, (xyz, rgb, label) in enumerate(frames, start=1):
            if scatter is not None:
                scatter.remove()

            if rgb is not None:
                colors = rgb
            else:
                z_norm = (xyz[:, 2] - xyz[:, 2].min()) / (xyz[:, 2].ptp() + 1e-6)
                colors = plt.cm.viridis(z_norm)

            scatter = ax.scatter(
                xyz[:, 0],
                xyz[:, 1],
                xyz[:, 2],
                s=0.5,
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
    frames = _prepare_frames(ply_paths, args.max_points)
    _render_sequence(frames, args.save, args.fps, args.title, args.elev, args.azim)
    print(f"Saved preview to {args.save} ({len(frames)} frame(s)).")


if __name__ == "__main__":
    main()
