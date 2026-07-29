from __future__ import annotations

import glob
import json
import math
import os
import queue
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk


DEFAULT_FPS = 1.0
DEFAULT_FOV = 90.0
DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080
DEFAULT_PARALLELISM = max(1, min(4, os.cpu_count() or 1))
MIN_WINDOW_FLOOR_WIDTH = 480
MIN_WINDOW_FLOOR_HEIGHT = 240
# プレビュー領域はウィンドウに追従して伸縮するため、ウィジェットの要求サイズは
# 小さくしておき、実際の表示サイズは <Configure> で決める。
PREVIEW_SURFACE_MIN_WIDTH = 320
PREVIEW_SURFACE_MIN_HEIGHT = 180
DEFAULT_SPHERE_CANVAS_SIZE = 140
MIN_SNAPSHOT_SIZE = 16
WINDOW_MAX_SCREEN_WIDTH_RATIO = 0.95
WINDOW_MAX_SCREEN_HEIGHT_RATIO = 0.92
PREFERRED_WINDOW_WIDTH = 1440
PREFERRED_WINDOW_HEIGHT = 1020
PREVIEW_PROXY_FPS_LIMIT = 15.0
SETTINGS_FILE_NAME = ".insta360_frame_extractor_gui.settings.json"
SEGFORMER_MODEL_ID = "nvidia/segformer-b0-finetuned-ade-512-512"
MASK_CATEGORY_LABEL_TOKENS: dict[str, tuple[str, ...]] = {
    "sky": ("sky",),
    "person": ("person",),
    "car": ("car",),
    "tree": ("tree",),
}
MASK_CATEGORY_DISPLAY_NAMES: dict[str, str] = {
    "sky": "空",
    "person": "人",
    "car": "車",
    "tree": "木",
}
MASK_BATCH_SIZE_LIMIT = 8
SETTINGS_COLUMN_COUNT = 8
DEFAULT_MASK_DETAIL_LEVEL = "標準"
DEFAULT_MASK_CONFIDENCE_THRESHOLD = 0.7
MASK_DETAIL_PRESETS: dict[str, dict[str, int | bool]] = {
    "標準": {
        "prioritize_detail": False,
        "tile_size": 0,
        "overlap": 0,
    },
    "高": {
        "prioritize_detail": True,
        "tile_size": 1536,
        "overlap": 192,
    },
    "最高": {
        "prioritize_detail": True,
        "tile_size": 1024,
        "overlap": 256,
    },
}
PREVIEW_COLORS = (
    "#38bdf8",
    "#fb7185",
    "#f59e0b",
    "#34d399",
    "#a78bfa",
    "#f97316",
    "#2dd4bf",
    "#e879f9",
)


@dataclass(frozen=True)
class Direction:
    yaw: float
    pitch: float


@dataclass(frozen=True)
class VideoMetadata:
    width: int
    height: int
    duration_seconds: float | None


@dataclass(frozen=True)
class ExtractedImageJob:
    image_path: Path
    mask_path: Path


@dataclass
class DirectionSet:
    name: str
    directions: list[Direction]


def default_settings_path() -> Path:
    return Path(__file__).resolve().with_name(SETTINGS_FILE_NAME)


def get_cv2() -> object:
    import cv2

    return cv2


def get_vlc() -> object:
    vlc_dir = Path(r"C:\Program Files\VideoLAN\VLC")
    if vlc_dir.is_dir() and hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(str(vlc_dir))
        except OSError:
            pass

    import vlc

    return vlc


def normalize_angle(value: float) -> float:
    normalized = ((value + 180.0) % 360.0) - 180.0
    if normalized == -180.0 and value > 0:
        return 180.0
    return normalized


def parse_positive_float(value: str, label: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{label}は数値で入力してください。") from exc

    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{label}は0より大きい値を指定してください。")
    return parsed


def parse_positive_int(value: str, label: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{label}は整数で入力してください。") from exc

    if parsed <= 0:
        raise ValueError(f"{label}は1以上を指定してください。")
    return parsed


def parse_angle(value: str, label: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{label}は数値で入力してください。") from exc

    if not math.isfinite(parsed):
        raise ValueError(f"{label}は有効な数値で入力してください。")
    return normalize_angle(parsed)


def safe_positive_float(value: str) -> float | None:
    try:
        parsed = float(value)
    except ValueError:
        return None

    if not math.isfinite(parsed) or parsed <= 0:
        return None
    return parsed


def parse_probability_threshold(value: str, label: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{label}は0より大きく1以下の数値で入力してください。") from exc

    # 0 を許すと「確率 >= 0」が常に真になり全面除外マスクになるため下限は開区間。
    if not math.isfinite(parsed) or not 0.0 < parsed <= 1.0:
        raise ValueError(f"{label}は0より大きく1以下の範囲で指定してください。")
    return parsed


def safe_positive_int(value: str) -> int | None:
    try:
        parsed = int(value)
    except ValueError:
        return None

    if parsed <= 0:
        return None
    return parsed


def direction_aware_float(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text or "0"


def direction_index_width(direction_count: int) -> int:
    return max(2, len(str(max(direction_count - 1, 0))))


def build_output_pattern(
    output_dir: Path,
    video_stem: str,
    direction_idx: int,
    direction_count: int,
) -> Path:
    width = direction_index_width(direction_count)
    return output_dir / f"{video_stem}_%04d_{direction_idx:0{width}d}.jpg"


def compute_output_direction_index(
    direction_idx: int,
    direction_count: int,
    frame_index: int,
    reverse_on_odd_frames: bool = False,
) -> int:
    if reverse_on_odd_frames and frame_index % 2 == 1:
        return (direction_count - 1) - direction_idx
    return direction_idx


def build_output_image_path(
    output_dir: Path,
    video_stem: str,
    frame_index: int,
    direction_idx: int,
    direction_count: int,
    reverse_on_odd_frames: bool = False,
) -> Path:
    output_direction_idx = compute_output_direction_index(
        direction_idx,
        direction_count,
        frame_index,
        reverse_on_odd_frames=reverse_on_odd_frames,
    )
    width = direction_index_width(direction_count)
    return output_dir / f"{video_stem}_{frame_index:04d}_{output_direction_idx:0{width}d}.jpg"


def build_mask_output_path(image_path: Path) -> Path:
    return image_path.with_name(image_path.name + ".mask.png")


def build_extracted_image_glob(video_stem: str | None) -> str:
    """抽出済み JPG を拾う glob パターン。動画名の `[` `]` `*` `?` を無効化する。"""
    if not video_stem:
        return "*.jpg"
    return f"{glob.escape(video_stem)}_*.jpg"


def resolve_mask_detail_preset(level_name: str) -> dict[str, int | bool]:
    return MASK_DETAIL_PRESETS.get(level_name, MASK_DETAIL_PRESETS[DEFAULT_MASK_DETAIL_LEVEL])


def collect_segformer_excluded_label_groups(id2label: dict[int, str]) -> dict[str, set[int]]:
    label_groups = {category: set() for category in MASK_CATEGORY_LABEL_TOKENS}
    for label_id, raw_label in id2label.items():
        normalized = raw_label.lower().replace(";", " ").replace(",", " ")
        tokens = {token.strip() for token in normalized.split() if token.strip()}
        for category, category_tokens in MASK_CATEGORY_LABEL_TOKENS.items():
            if any(token in tokens for token in category_tokens):
                label_groups[category].add(int(label_id))
    return label_groups


def collect_segformer_excluded_label_ids(id2label: dict[int, str]) -> set[int]:
    label_groups = collect_segformer_excluded_label_groups(id2label)
    excluded_label_ids: set[int] = set()
    for label_ids in label_groups.values():
        excluded_label_ids.update(label_ids)
    return excluded_label_ids


def build_binary_usage_mask(
    excluded_probability: object,
    confidence_threshold: float,
) -> object:
    import numpy as np

    probability_array = np.asarray(excluded_probability, dtype=np.float32)
    return np.where(probability_array >= confidence_threshold, 0, 255).astype(np.uint8)


def mask_contains_excluded_region(binary_mask: object) -> bool:
    import numpy as np

    mask_array = np.asarray(binary_mask, dtype=np.uint8)
    return bool(np.any(mask_array == 0))


def build_tile_starts(full_size: int, tile_size: int, overlap: int) -> list[int]:
    effective_tile_size = max(1, min(tile_size, full_size))
    stride = max(1, effective_tile_size - overlap)

    starts = list(range(0, max(full_size - effective_tile_size, 0) + 1, stride))
    last_start = max(full_size - effective_tile_size, 0)
    if not starts or starts[-1] != last_start:
        starts.append(last_start)
    return starts


def build_overlapping_tile_regions(
    image_width: int,
    image_height: int,
    tile_size: int,
    overlap: int,
) -> list[tuple[int, int, int, int]]:
    tile_width = max(1, min(tile_size, image_width))
    tile_height = max(1, min(tile_size, image_height))
    x_starts = build_tile_starts(image_width, tile_width, overlap)
    y_starts = build_tile_starts(image_height, tile_height, overlap)

    return [
        (
            left,
            top,
            min(left + tile_width, image_width),
            min(top + tile_height, image_height),
        )
        for top in y_starts
        for left in x_starts
    ]


def compute_vertical_fov(horizontal_fov: float, width: int, height: int) -> float:
    aspect_ratio = height / width
    return math.degrees(2.0 * math.atan(math.tan(math.radians(horizontal_fov) / 2.0) * aspect_ratio))


def build_filter(
    direction: Direction,
    fps: float,
    fov: float,
    width: int,
    height: int,
    reverse_video: bool = False,
) -> str:
    vertical_fov = compute_vertical_fov(fov, width, height)
    reverse_prefix = "reverse," if reverse_video else ""
    return (
        "fps={fps},"
        "{reverse_prefix}"
        "v360=input=equirect:output=flat:interp=cubic:w={width}:h={height}:"
        "h_fov={fov}:v_fov={v_fov}:yaw={yaw}:pitch={pitch},"
        "setsar=1"
    ).format(
        fps=direction_aware_float(fps),
        reverse_prefix=reverse_prefix,
        width=width,
        height=height,
        fov=direction_aware_float(fov),
        v_fov=direction_aware_float(vertical_fov),
        yaw=direction_aware_float(direction.yaw),
        pitch=direction_aware_float(direction.pitch),
    )


def build_filter_chain(
    direction: Direction,
    fps: float,
    fov: float,
    width: int,
    height: int,
    use_gpu_decode: bool,
    reverse_video: bool = False,
) -> str:
    filter_chain = build_filter(direction, fps, fov, width, height, reverse_video=reverse_video)
    if use_gpu_decode:
        return f"hwdownload,format=nv12,{filter_chain}"
    return filter_chain


def build_ffmpeg_command(
    ffmpeg_path: str,
    input_video: Path,
    output_dir: Path,
    video_stem: str,
    direction: Direction,
    direction_idx: int,
    direction_count: int,
    fps: float,
    fov: float,
    width: int,
    height: int,
    use_gpu_decode: bool = False,
    reverse_video: bool = False,
    output_pattern_override: Path | None = None,
) -> list[str]:
    filter_chain = build_filter_chain(
        direction,
        fps,
        fov,
        width,
        height,
        use_gpu_decode,
        reverse_video=reverse_video,
    )
    output_pattern = output_pattern_override or build_output_pattern(
        output_dir,
        video_stem,
        direction_idx,
        direction_count,
    )

    command = [
        ffmpeg_path,
        "-hide_banner",
        "-y",
    ]

    if use_gpu_decode:
        command.extend(["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"])

    command.extend(
        [
            "-i",
            str(input_video),
            "-an",
            "-vf",
            filter_chain,
            "-q:v",
            "2",
            "-start_number",
            "0",
            str(output_pattern),
        ]
    )
    return command


def compute_focal_length_35mm(horizontal_fov: float, width: int, height: int) -> float:
    """v360 の flat 出力に対応する 35mm 換算焦点距離を返す。

    v360 の flat 出力は理想ピンホール投影そのもので、レンズ歪みが無く、主点は画像中心、
    画素は正方 (`compute_vertical_fov` が水平画角とアスペクト比から v_fov を導くため
    fx == fy になる)。したがって焦点距離は推定するまでもなく水平画角から厳密に決まる。

    RealityScan / RealityCapture は 35mm 換算焦点距離を画像の長辺基準で扱うため、
    f_px = (width / 2) / tan(h_fov / 2) を長辺で正規化して 36mm を掛ける。
    """
    half_angle = math.tan(math.radians(horizontal_fov) / 2.0)
    if half_angle <= 0.0:
        raise ValueError("画角は 0 度より大きく 180 度未満で指定してください。")
    focal_length_pixels = (width / 2.0) / half_angle
    return 36.0 * focal_length_pixels / max(width, height)


def build_xmp_sidecar_path(image_path: Path) -> Path:
    """RealityScan は `<画像名>.xmp` を同じフォルダから読み込む。"""
    return image_path.with_suffix(".xmp")


def build_xmp_document(
    focal_length_35mm: float,
    calibration_group: int = 0,
    distortion_group: int = 0,
    in_texturing: bool = True,
    in_meshing: bool = True,
) -> str:
    """RealityScan / RealityCapture 用の XMP サイドカーを組み立てる。

    姿勢 (Rotation / Position) は書かない。回転行列の座標系規約を実機で確認できていないため、
    誤った姿勢事前情報はアライメントを助けるどころか破壊しうる。
    一方でキャリブレーションは v360 の出力仕様から厳密に確定するので、そこだけを渡す。

    `CalibrationGroup` / `DistortionGroup` を全画像で共有させることで、
    「これらは全て同一の合成カメラで撮られた」と RealityScan に伝えられる。
    これによりキャリブレーションの推定が 1 グループに集約され、収束が速く安定する。
    """
    return (
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">\n'
        '  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
        '    <rdf:Description xmlns:xcr="http://www.capturingreality.com/ns/xcr/1.1#"\n'
        '      xcr:Version="2"\n'
        '      xcr:DistortionModel="division"\n'
        f'      xcr:FocalLength35mm="{focal_length_35mm:.9f}"\n'
        '      xcr:CalibrationPrior="initial"\n'
        f'      xcr:CalibrationGroup="{calibration_group}"\n'
        f'      xcr:DistortionGroup="{distortion_group}"\n'
        f'      xcr:InTexturing="{1 if in_texturing else 0}"\n'
        f'      xcr:InMeshing="{1 if in_meshing else 0}"/>\n'
        '  </rdf:RDF>\n'
        '</x:xmpmeta>\n'
    )


def write_xmp_sidecar(
    image_path: Path,
    focal_length_35mm: float,
    calibration_group: int = 0,
    distortion_group: int = 0,
) -> Path:
    sidecar_path = build_xmp_sidecar_path(image_path)
    sidecar_path.write_text(
        build_xmp_document(
            focal_length_35mm,
            calibration_group=calibration_group,
            distortion_group=distortion_group,
        ),
        encoding="utf-8",
    )
    return sidecar_path


def build_single_pass_filter_complex(
    directions: list[Direction],
    fps: float,
    fov: float,
    width: int,
    height: int,
    use_gpu_decode: bool,
    reverse_video: bool = False,
) -> tuple[str, list[str]]:
    """全方向を1回のデコードで切り出す filter_complex を組み立てる。

    方向ごとに ffmpeg を起動すると、方向数だけ入力動画を丸ごとデコードし直すことになる。
    デコードは v360 変換より遥かに重いため、`split` で分岐して 1 パスにまとめる。
    出力画素は方向ごとに起動した場合とバイト単位で一致する。
    """
    if not directions:
        raise ValueError("方向が指定されていません。")

    vertical_fov = compute_vertical_fov(fov, width, height)
    output_labels = [f"o{index}" for index in range(len(directions))]
    split_labels = "".join(f"[s{index}]" for index in range(len(directions)))

    head = "[0:v]"
    if use_gpu_decode:
        head += "hwdownload,format=nv12,"
    head += f"fps={direction_aware_float(fps)},"
    if reverse_video:
        head += "reverse,"
    head += f"split={len(directions)}{split_labels}"

    branches = [
        (
            "[s{index}]v360=input=equirect:output=flat:interp=cubic:w={width}:h={height}:"
            "h_fov={fov}:v_fov={v_fov}:yaw={yaw}:pitch={pitch},setsar=1[{label}]"
        ).format(
            index=index,
            width=width,
            height=height,
            fov=direction_aware_float(fov),
            v_fov=direction_aware_float(vertical_fov),
            yaw=direction_aware_float(direction.yaw),
            pitch=direction_aware_float(direction.pitch),
            label=output_labels[index],
        )
        for index, direction in enumerate(directions)
    ]

    return ";".join([head, *branches]), output_labels


def build_single_pass_ffmpeg_command(
    ffmpeg_path: str,
    input_video: Path,
    output_patterns: list[Path],
    directions: list[Direction],
    fps: float,
    fov: float,
    width: int,
    height: int,
    use_gpu_decode: bool = False,
    reverse_video: bool = False,
) -> list[str]:
    if len(output_patterns) != len(directions):
        raise ValueError("出力パターン数と方向数が一致しません。")

    filter_complex, output_labels = build_single_pass_filter_complex(
        directions,
        fps,
        fov,
        width,
        height,
        use_gpu_decode,
        reverse_video=reverse_video,
    )

    command = [
        ffmpeg_path,
        "-hide_banner",
        "-y",
    ]

    if use_gpu_decode:
        command.extend(["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"])

    command.extend(["-i", str(input_video), "-an", "-filter_complex", filter_complex])

    for label, output_pattern in zip(output_labels, output_patterns):
        command.extend(
            [
                "-map",
                f"[{label}]",
                "-q:v",
                "2",
                "-start_number",
                "0",
                str(output_pattern),
            ]
        )

    return command


def generate_ring_directions(count: int, pitch: float) -> list[Direction]:
    return [Direction(yaw=normalize_angle(index * 360.0 / count), pitch=pitch) for index in range(count)]


def format_duration(duration_seconds: float | None) -> str:
    if duration_seconds is None:
        return "unknown"

    rounded = max(0, int(round(duration_seconds)))
    hours, remainder = divmod(rounded, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def format_timestamp(seconds_value: float) -> str:
    clamped_seconds = max(0.0, seconds_value)
    total_seconds = int(clamped_seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def compute_preview_box(
    source_width: int,
    source_height: int,
    canvas_width: int,
    canvas_height: int,
) -> tuple[float, float, float, float]:
    # 極端に小さい / 壊れたサイズでも 0 除算せずに描画できる値を返す。
    safe_source_width = max(1, int(source_width))
    safe_source_height = max(1, int(source_height))
    safe_canvas_width = max(1, int(canvas_width))
    safe_canvas_height = max(1, int(canvas_height))

    scale = min(safe_canvas_width / safe_source_width, safe_canvas_height / safe_source_height)
    display_width = safe_source_width * scale
    display_height = safe_source_height * scale
    offset_x = (safe_canvas_width - display_width) / 2.0
    offset_y = (safe_canvas_height - display_height) / 2.0
    return offset_x, offset_y, display_width, display_height


def fit_size_within_bounds(
    source_width: int,
    source_height: int,
    max_width: int,
    max_height: int,
    allow_upscale: bool = False,
) -> tuple[int, int]:
    safe_source_width = max(1, int(source_width))
    safe_source_height = max(1, int(source_height))
    safe_max_width = max(1, int(max_width))
    safe_max_height = max(1, int(max_height))

    scale = min(safe_max_width / safe_source_width, safe_max_height / safe_source_height)
    if not allow_upscale:
        scale = min(scale, 1.0)

    target_width = max(1, min(safe_max_width, int(round(safe_source_width * scale))))
    target_height = max(1, min(safe_max_height, int(round(safe_source_height * scale))))
    return target_width, target_height


def vector_length(vector: tuple[float, float, float]) -> float:
    x, y, z = vector
    return math.sqrt((x * x) + (y * y) + (z * z))


def normalize_vector(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    length = vector_length(vector)
    if length == 0:
        raise ValueError("ゼロベクトルは正規化できません。")
    x, y, z = vector
    return x / length, y / length, z / length


def rotate_local_vector(
    vector: tuple[float, float, float],
    yaw_degrees: float,
    pitch_degrees: float,
) -> tuple[float, float, float]:
    return normalize_vector(rotate_local_point(vector, yaw_degrees, pitch_degrees))


def rotate_local_point(
    vector: tuple[float, float, float],
    yaw_degrees: float,
    pitch_degrees: float,
) -> tuple[float, float, float]:
    yaw_radians = math.radians(yaw_degrees)
    pitch_radians = math.radians(pitch_degrees)

    x, y, z = vector

    cos_pitch = math.cos(pitch_radians)
    sin_pitch = math.sin(pitch_radians)
    x_pitch = x
    y_pitch = (y * cos_pitch) + (z * sin_pitch)
    z_pitch = (-y * sin_pitch) + (z * cos_pitch)

    cos_yaw = math.cos(yaw_radians)
    sin_yaw = math.sin(yaw_radians)
    x_yaw = (x_pitch * cos_yaw) + (z_pitch * sin_yaw)
    y_yaw = y_pitch
    z_yaw = (-x_pitch * sin_yaw) + (z_pitch * cos_yaw)

    return x_yaw, y_yaw, z_yaw


def vector_to_equirect_point(
    vector: tuple[float, float, float],
    map_width: int,
    map_height: int,
) -> tuple[float, float]:
    x, y, z = normalize_vector(vector)
    longitude = math.degrees(math.atan2(x, z))
    latitude = math.degrees(math.asin(max(-1.0, min(1.0, y))))
    point_x = ((longitude + 180.0) / 360.0) * map_width
    point_y = ((90.0 - latitude) / 180.0) * map_height
    return point_x, point_y


def direction_to_equirect_point(
    direction: Direction,
    map_width: int,
    map_height: int,
) -> tuple[float, float]:
    center_vector = rotate_local_vector((0.0, 0.0, 1.0), direction.yaw, direction.pitch)
    return vector_to_equirect_point(center_vector, map_width, map_height)


def rotate_vector_y(vector: tuple[float, float, float], degrees_value: float) -> tuple[float, float, float]:
    radians_value = math.radians(degrees_value)
    cosine = math.cos(radians_value)
    sine = math.sin(radians_value)
    x_value, y_value, z_value = vector
    return (
        (x_value * cosine) + (z_value * sine),
        y_value,
        (-x_value * sine) + (z_value * cosine),
    )


def rotate_vector_x(vector: tuple[float, float, float], degrees_value: float) -> tuple[float, float, float]:
    radians_value = math.radians(degrees_value)
    cosine = math.cos(radians_value)
    sine = math.sin(radians_value)
    x_value, y_value, z_value = vector
    return (
        x_value,
        (y_value * cosine) - (z_value * sine),
        (y_value * sine) + (z_value * cosine),
    )


def apply_view_rotation(
    vector: tuple[float, float, float],
    view_yaw: float,
    view_pitch: float,
) -> tuple[float, float, float]:
    rotated = rotate_vector_y(vector, view_yaw)
    rotated = rotate_vector_x(rotated, view_pitch)
    return rotated


def build_viewport_perimeter_samples(points_per_side: int = 18) -> list[tuple[float, float]]:
    points_per_side = max(2, points_per_side)
    samples: list[tuple[float, float]] = []

    for index in range(points_per_side):
        ratio = index / (points_per_side - 1)
        samples.append((-1.0 + (2.0 * ratio), 1.0))

    for index in range(1, points_per_side):
        ratio = index / (points_per_side - 1)
        samples.append((1.0, 1.0 - (2.0 * ratio)))

    for index in range(1, points_per_side):
        ratio = index / (points_per_side - 1)
        samples.append((1.0 - (2.0 * ratio), -1.0))

    for index in range(1, points_per_side - 1):
        ratio = index / (points_per_side - 1)
        samples.append((-1.0, -1.0 + (2.0 * ratio)))

    samples.append(samples[0])
    return samples


def unwrap_polyline(
    points: list[tuple[float, float]],
    map_width: int,
) -> list[tuple[float, float]]:
    if not points:
        return []

    unwrapped = [points[0]]
    offset = 0.0
    previous_x = points[0][0]

    for point in points[1:]:
        point_x, point_y = point
        adjusted_x = point_x + offset
        delta_x = adjusted_x - previous_x

        if delta_x > (map_width / 2.0):
            offset -= map_width
        elif delta_x < -(map_width / 2.0):
            offset += map_width

        adjusted_x = point_x + offset
        unwrapped.append((adjusted_x, point_y))
        previous_x = adjusted_x

    return unwrapped


def build_footprint_segments(
    direction: Direction,
    horizontal_fov: float,
    aspect_ratio: float,
    map_width: int,
    map_height: int,
    points_per_side: int = 18,
) -> list[list[tuple[float, float]]]:
    if aspect_ratio <= 0:
        return []

    clamped_fov = max(1.0, min(horizontal_fov, 179.0))
    tangent_horizontal = math.tan(math.radians(clamped_fov) / 2.0)
    tangent_vertical = tangent_horizontal / aspect_ratio

    points: list[tuple[float, float]] = []
    for sample_x, sample_y in build_viewport_perimeter_samples(points_per_side):
        local_vector = normalize_vector((sample_x * tangent_horizontal, sample_y * tangent_vertical, 1.0))
        world_vector = rotate_local_vector(local_vector, direction.yaw, direction.pitch)
        points.append(vector_to_equirect_point(world_vector, map_width, map_height))

    return [unwrap_polyline(points, map_width)]


def map_preview_point(
    point: tuple[float, float],
    preview_box: tuple[float, float, float, float],
    source_width: int,
    source_height: int,
) -> tuple[float, float]:
    offset_x, offset_y, display_width, display_height = preview_box
    point_x, point_y = point
    canvas_x = offset_x + ((point_x / source_width) * display_width)
    canvas_y = offset_y + ((point_y / source_height) * display_height)
    return canvas_x, canvas_y


def pick_preview_seek_seconds(duration_seconds: float | None) -> float:
    if duration_seconds is None:
        return 0.0
    if duration_seconds <= 1.0:
        return 0.0
    return min(1.0, duration_seconds / 2.0)


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=creationflags,
        check=False,
    )


def detect_cuda_hwaccel(ffmpeg_path: str | None) -> bool:
    if not ffmpeg_path:
        return False

    result = run_command([ffmpeg_path, "-hide_banner", "-hwaccels"])
    if result.returncode != 0:
        return False
    return "cuda" in result.stdout.split()


def probe_video_metadata(video_path: Path, ffprobe_path: str) -> VideoMetadata:
    result = run_command(
        [
            ffprobe_path,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height:format=duration",
            "-of",
            "json",
            str(video_path),
        ]
    )

    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "ffprobe failed"
        raise RuntimeError(f"動画情報の取得に失敗しました: {detail}")

    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    if not streams:
        raise RuntimeError("動画ストリームが見つかりませんでした。")

    stream = streams[0]
    try:
        width = int(stream["width"])
        height = int(stream["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("動画の解像度を取得できませんでした。") from exc

    if width <= 0 or height <= 0:
        raise RuntimeError(f"動画の解像度が不正です: {width}x{height}")

    duration_text = payload.get("format", {}).get("duration")
    try:
        duration_seconds = float(duration_text) if duration_text is not None else None
    except (TypeError, ValueError):
        duration_seconds = None

    if duration_seconds is not None and (not math.isfinite(duration_seconds) or duration_seconds <= 0.0):
        duration_seconds = None

    return VideoMetadata(width=width, height=height, duration_seconds=duration_seconds)


def extract_preview_image(
    video_path: Path,
    ffmpeg_path: str,
    metadata: VideoMetadata,
    preview_width: int,
    preview_height: int,
) -> Path:
    with tempfile.NamedTemporaryFile(prefix="insta360_preview_", suffix=".png", delete=False) as temporary:
        preview_path = Path(temporary.name)

    seek_seconds = pick_preview_seek_seconds(metadata.duration_seconds)
    filter_chain = f"scale={preview_width}:{preview_height}:flags=lanczos,setsar=1"

    command = [
        ffmpeg_path,
        "-hide_banner",
        "-y",
        "-ss",
        direction_aware_float(seek_seconds),
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-vf",
        filter_chain,
        str(preview_path),
    ]

    result = run_command(command)
    if result.returncode != 0:
        preview_path.unlink(missing_ok=True)
        detail = result.stderr.strip() or result.stdout.strip() or "ffmpeg failed"
        raise RuntimeError(f"サムネイル生成に失敗しました: {detail}")

    return preview_path


def extract_preview_proxy_video(
    video_path: Path,
    ffmpeg_path: str,
    preview_width: int,
    preview_height: int,
    fps_limit: float = PREVIEW_PROXY_FPS_LIMIT,
) -> Path:
    with tempfile.NamedTemporaryFile(prefix="insta360_preview_proxy_", suffix=".avi", delete=False) as temporary:
        proxy_path = Path(temporary.name)

    filter_chain = (
        f"scale={preview_width}:{preview_height}:flags=fast_bilinear,"
        f"fps={direction_aware_float(fps_limit)},setsar=1"
    )
    command = [
        ffmpeg_path,
        "-hide_banner",
        "-y",
        "-i",
        str(video_path),
        "-an",
        "-vf",
        filter_chain,
        "-c:v",
        "mjpeg",
        "-q:v",
        "5",
        str(proxy_path),
    ]

    result = run_command(command)
    if result.returncode != 0:
        proxy_path.unlink(missing_ok=True)
        detail = result.stderr.strip() or result.stdout.strip() or "ffmpeg failed"
        raise RuntimeError(f"プレビュー動画生成に失敗しました: {detail}")

    return proxy_path


class SegFormerMaskGenerator:
    def __init__(self, model_id: str = SEGFORMER_MODEL_ID) -> None:
        try:
            import numpy as np
            import torch
            from PIL import Image
            from transformers import AutoModelForSemanticSegmentation, SegformerImageProcessor
        except ImportError as exc:
            raise RuntimeError(
                "SegFormer マスク生成には `transformers` が必要です。"
                " `python -m pip install transformers` を実行してください。"
            ) from exc

        self.np = np
        self.torch = torch
        self.Image = Image
        self.model_id = model_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_processor = SegformerImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(model_id).to(self.device)
        self.model.eval()

        raw_id2label = getattr(self.model.config, "id2label", {})
        self.id2label = {int(label_id): str(label_name) for label_id, label_name in raw_id2label.items()}
        self.excluded_label_groups = collect_segformer_excluded_label_groups(self.id2label)
        self.excluded_label_ids = collect_segformer_excluded_label_ids(self.id2label)
        if not self.excluded_label_ids:
            raise RuntimeError("SegFormer のラベルから sky / person / car / tree を解決できませんでした。")

    def device_label(self) -> str:
        return str(self.device)

    def resolve_selected_label_ids(self, selected_categories: list[str]) -> set[int]:
        selected_label_ids: set[int] = set()
        for category in selected_categories:
            selected_label_ids.update(self.excluded_label_groups.get(category, set()))
        return selected_label_ids

    def generate_masks(
        self,
        image_jobs: list[ExtractedImageJob],
        batch_size: int,
        stop_requested: threading.Event,
        detail_level: str = DEFAULT_MASK_DETAIL_LEVEL,
        confidence_threshold: float = DEFAULT_MASK_CONFIDENCE_THRESHOLD,
        selected_categories: list[str] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        total = len(image_jobs)
        if total == 0:
            return

        batch_size = max(1, min(batch_size, MASK_BATCH_SIZE_LIMIT))
        detail_preset = resolve_mask_detail_preset(detail_level)
        prioritize_detail = bool(detail_preset["prioritize_detail"])
        effective_categories = selected_categories or list(MASK_CATEGORY_LABEL_TOKENS.keys())
        selected_label_ids = self.resolve_selected_label_ids(effective_categories)
        if not selected_label_ids:
            raise RuntimeError("選択されたマスク対象に対応する SegFormer ラベルが見つかりませんでした。")

        if prioritize_detail:
            for image_index, image_job in enumerate(image_jobs, start=1):
                if stop_requested.is_set():
                    return

                self._generate_mask_high_detail(
                    image_job=image_job,
                    batch_size=batch_size,
                    stop_requested=stop_requested,
                    tile_size=int(detail_preset["tile_size"]),
                    overlap=int(detail_preset["overlap"]),
                    confidence_threshold=confidence_threshold,
                    selected_label_ids=selected_label_ids,
                )
                if progress_callback is not None:
                    progress_callback(image_index, total)
            return

        for batch_start in range(0, total, batch_size):
            if stop_requested.is_set():
                return

            batch_jobs = image_jobs[batch_start : batch_start + batch_size]
            batch_images: list[object] = []
            batch_sizes: list[tuple[int, int]] = []
            for image_job in batch_jobs:
                with self.Image.open(image_job.image_path) as image:
                    rgb_image = image.convert("RGB")
                    batch_sizes.append(rgb_image.size)
                    batch_images.append(rgb_image)

            resized_masks = self._predict_resized_masks(
                batch_images,
                batch_sizes,
                confidence_threshold,
                selected_label_ids,
            )
            for image_job, resized_mask in zip(batch_jobs, resized_masks):
                self._save_mask_array(resized_mask, image_job.mask_path)

            if progress_callback is not None:
                progress_callback(min(batch_start + len(batch_jobs), total), total)

    def _predict_resized_masks(
        self,
        batch_images: list[object],
        batch_sizes: list[tuple[int, int]],
        confidence_threshold: float,
        selected_label_ids: set[int],
    ) -> list[object]:
        inputs = self.image_processor(images=batch_images, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with self.torch.no_grad():
            outputs = self.model(**inputs)

        excluded_indices = sorted(selected_label_ids)
        probabilities = self.torch.nn.functional.softmax(outputs.logits, dim=1)
        excluded_probability_maps = probabilities[:, excluded_indices, :, :].sum(dim=1, keepdim=True)

        unique_sizes = set(batch_sizes)
        if len(unique_sizes) == 1:
            target_width, target_height = batch_sizes[0]
            resized_probability_maps = self.torch.nn.functional.interpolate(
                excluded_probability_maps,
                size=(target_height, target_width),
                mode="bilinear",
                align_corners=False,
            )
            resized_probabilities = resized_probability_maps[:, 0].cpu().numpy()
            return [
                build_binary_usage_mask(resized_probability, confidence_threshold)
                for resized_probability in resized_probabilities
            ]

        binary_masks: list[object] = []
        for batch_index, target_size in enumerate(batch_sizes):
            target_width, target_height = target_size
            resized_probability_map = self.torch.nn.functional.interpolate(
                excluded_probability_maps[batch_index : batch_index + 1],
                size=(target_height, target_width),
                mode="bilinear",
                align_corners=False,
            )[0, 0].cpu().numpy()
            binary_masks.append(build_binary_usage_mask(resized_probability_map, confidence_threshold))
        return binary_masks

    def _generate_mask_high_detail(
        self,
        image_job: ExtractedImageJob,
        batch_size: int,
        stop_requested: threading.Event,
        tile_size: int,
        overlap: int,
        confidence_threshold: float,
        selected_label_ids: set[int],
    ) -> None:
        with self.Image.open(image_job.image_path) as image:
            rgb_image = image.convert("RGB")
            image_width, image_height = rgb_image.size

            tile_regions = build_overlapping_tile_regions(
                image_width,
                image_height,
                tile_size,
                overlap,
            )
            full_prediction = self.np.zeros((image_height, image_width), dtype=self.np.uint8)
            overlap_margin = overlap // 2

            for batch_start in range(0, len(tile_regions), batch_size):
                if stop_requested.is_set():
                    return

                region_batch = tile_regions[batch_start : batch_start + batch_size]
                batch_images: list[object] = []
                batch_sizes: list[tuple[int, int]] = []

                for left, top, right, bottom in region_batch:
                    tile_image = rgb_image.crop((left, top, right, bottom))
                    batch_images.append(tile_image)
                    batch_sizes.append(tile_image.size)

                resized_masks = self._predict_resized_masks(
                    batch_images,
                    batch_sizes,
                    confidence_threshold,
                    selected_label_ids,
                )
                for (left, top, right, bottom), mask_array in zip(region_batch, resized_masks):
                    tile_width = right - left
                    tile_height = bottom - top
                    horizontal_margin = min(overlap_margin, tile_width // 2)
                    vertical_margin = min(overlap_margin, tile_height // 2)

                    inner_left = 0 if left == 0 else horizontal_margin
                    inner_top = 0 if top == 0 else vertical_margin
                    inner_right = tile_width if right == image_width else max(inner_left + 1, tile_width - horizontal_margin)
                    inner_bottom = tile_height if bottom == image_height else max(inner_top + 1, tile_height - vertical_margin)

                    full_prediction[
                        top + inner_top : top + inner_bottom,
                        left + inner_left : left + inner_right,
                    ] = mask_array[inner_top:inner_bottom, inner_left:inner_right]

            self._save_mask_array(full_prediction, image_job.mask_path)

    def _save_mask_array(self, binary_mask: object, mask_path: Path) -> None:
        mask_array = self.np.asarray(binary_mask, dtype=self.np.uint8)
        if not mask_contains_excluded_region(mask_array):
            mask_path.unlink(missing_ok=True)
            return
        self.Image.fromarray(mask_array, mode="L").save(mask_path)


class Insta360ExtractorApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        # 最小サイズは UI 構築後に実際の要求サイズから決めるため、ここでは暫定値。
        self.min_window_width = MIN_WINDOW_FLOOR_WIDTH
        self.min_window_height = MIN_WINDOW_FLOOR_HEIGHT
        self.root.title("Insta360 Frame Extractor")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.ffmpeg_path = shutil.which("ffmpeg")
        self.cuda_available = detect_cuda_hwaccel(self.ffmpeg_path)
        self.settings_path = default_settings_path()

        self.input_path_var = tk.StringVar()
        self.output_dir_var = tk.StringVar()
        self.fps_var = tk.StringVar(value=str(DEFAULT_FPS))
        self.fov_var = tk.StringVar(value=str(DEFAULT_FOV))
        self.width_var = tk.StringVar(value=str(DEFAULT_WIDTH))
        self.height_var = tk.StringVar(value=str(DEFAULT_HEIGHT))
        self.parallelism_var = tk.StringVar(value=str(DEFAULT_PARALLELISM))
        self.mask_parallelism_var = tk.StringVar(value=str(DEFAULT_PARALLELISM))
        self.run_extract_var = tk.BooleanVar(value=True)
        self.use_gpu_var = tk.BooleanVar(value=self.cuda_available)
        self.reverse_var = tk.BooleanVar(value=False)
        self.single_pass_var = tk.BooleanVar(value=True)
        self.write_xmp_var = tk.BooleanVar(value=False)
        self.reverse_direction_index_on_odd_var = tk.BooleanVar(value=False)
        self.generate_masks_var = tk.BooleanVar(value=False)
        self.mask_detail_level_var = tk.StringVar(value=DEFAULT_MASK_DETAIL_LEVEL)
        self.mask_confidence_threshold_var = tk.StringVar(value=direction_aware_float(DEFAULT_MASK_CONFIDENCE_THRESHOLD))
        self.mask_sky_var = tk.BooleanVar(value=True)
        self.mask_person_var = tk.BooleanVar(value=True)
        self.mask_car_var = tk.BooleanVar(value=True)
        self.mask_tree_var = tk.BooleanVar(value=True)
        self.yaw_var = tk.StringVar(value="0")
        self.pitch_var = tk.StringVar(value="0")
        self.ring_count_var = tk.StringVar(value="8")
        self.ring_pitch_var = tk.StringVar(value="0")
        self.status_var = tk.StringVar(value="入力を設定してください。")
        self.preview_info_var = tk.StringVar(value="動画を選択するとサムネイルを表示します。")
        self.preview_selection_var = tk.StringVar(value="方向を選択すると強調表示されます。")

        self.direction_sets: list[DirectionSet] = [DirectionSet(name="セット1", directions=[Direction(yaw=0.0, pitch=0.0)])]
        self.active_direction_set_index = 0
        self.directions: list[Direction] = self.direction_sets[0].directions
        # "log" / "status" / "error" は str、プレビュー関連は dict を載せる。
        self.log_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.worker_thread: threading.Thread | None = None
        self.stop_requested = threading.Event()
        self.process_lock = threading.Lock()
        self.current_processes: dict[int, subprocess.Popen[str]] = {}
        self.segformer_mask_generator: SegFormerMaskGenerator | None = None

        self.preview_photo: tk.PhotoImage | None = None
        self.preview_path: Path | None = None
        self.preview_metadata: VideoMetadata | None = None
        self.preview_box: tuple[float, float, float, float] | None = None
        self.preview_canvas_width = PREVIEW_SURFACE_MIN_WIDTH
        self.preview_canvas_height = PREVIEW_SURFACE_MIN_HEIGHT
        self.preview_canvas_offset_x = 0
        self.preview_canvas_offset_y = 0
        self.preview_last_frame_bgr: object | None = None
        self.preview_photo_from_snapshot = False
        self.preview_mode = "none"
        self.preview_capture: object | None = None
        self.preview_proxy_path: Path | None = None
        self.preview_current_video_path: Path | None = None
        self.preview_current_frame_index = 0
        self.preview_total_frames = 0
        self.preview_fps = 0.0
        self.preview_duration_seconds = 0.0
        self.preview_is_playing = False
        self.preview_playback_after_id: str | None = None
        self.preview_slider_active = False
        self.preview_slider_internal_update = False
        self.preview_seek_var = tk.DoubleVar(value=0.0)
        self.preview_time_var = tk.StringVar(value="00:00 / 00:00")
        self.preview_play_button_var = tk.StringVar(value="再生")
        self.preview_overlay_sync_var = tk.BooleanVar(value=False)
        self.preview_resize_after_id: str | None = None
        self.preview_seek_after_id: str | None = None
        self.preview_proxy_request_id = 0
        self.preview_proxy_preparing = False
        self.preview_vlc_module: object | None = None
        self.preview_vlc_instance: object | None = None
        self.preview_vlc_player: object | None = None
        self.preview_vlc_poll_after_id: str | None = None
        self.preview_snapshot_after_id: str | None = None
        self.suspend_preview_mode_reload = False
        self.sphere_canvas_size = DEFAULT_SPHERE_CANVAS_SIZE
        self.sphere_view_yaw = 0.0
        self.sphere_view_pitch = 0.0
        self.sphere_drag_last: tuple[int, int] | None = None

        self._build_ui()
        self._bind_preview_refresh()
        self._load_persisted_settings()
        self._refresh_direction_tabs()
        self._refresh_direction_table(select_index=0)
        self._apply_window_size_limits(initialize=True)
        self.log_drain_after_id: str | None = self.root.after(100, self._drain_log_queue)

    def _window_size_ceiling(self) -> tuple[int, int]:
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        return (
            max(MIN_WINDOW_FLOOR_WIDTH, int(screen_width * WINDOW_MAX_SCREEN_WIDTH_RATIO)),
            max(MIN_WINDOW_FLOOR_HEIGHT, int(screen_height * WINDOW_MAX_SCREEN_HEIGHT_RATIO)),
        )

    def _apply_window_size_limits(self, initialize: bool = False) -> None:
        """UI の実要求サイズから最小ウィンドウサイズを決める。

        最小サイズを実測から決めることで「縮めるとボタンが消える」状態を作れなくする。
        方向セットのタブが増えて必要幅が伸びた場合も追従できるよう、初回以降は
        最小サイズを広げる方向にだけ更新する。
        """
        self.root.update_idletasks()
        max_width, max_height = self._window_size_ceiling()

        required_width = min(self.root.winfo_reqwidth(), max_width)
        required_height = min(self.root.winfo_reqheight(), max_height)
        if initialize:
            self.min_window_width = required_width
            self.min_window_height = required_height
        else:
            self.min_window_width = max(self.min_window_width, required_width)
            self.min_window_height = max(self.min_window_height, required_height)
        self.root.minsize(self.min_window_width, self.min_window_height)

        if initialize:
            target_width = min(max(self.min_window_width, PREFERRED_WINDOW_WIDTH), max_width)
            target_height = min(max(self.min_window_height, PREFERRED_WINDOW_HEIGHT), max_height)
        else:
            # 既にユーザーが決めたサイズは尊重し、最小を下回る時だけ広げる。
            target_width = max(self.root.winfo_width(), self.min_window_width)
            target_height = max(self.root.winfo_height(), self.min_window_height)
            if target_width <= self.root.winfo_width() and target_height <= self.root.winfo_height():
                return
        self.root.geometry(f"{target_width}x{target_height}")

    def _build_ui(self) -> None:
        root_frame = ttk.Frame(self.root, padding=16)
        root_frame.pack(fill="both", expand=True)

        path_frame = ttk.LabelFrame(root_frame, text="入出力", padding=12)
        path_frame.pack(fill="x")
        path_frame.columnconfigure(1, weight=1)

        ttk.Label(path_frame, text="入力動画").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=6)
        ttk.Entry(path_frame, textvariable=self.input_path_var).grid(row=0, column=1, sticky="ew", pady=6)
        ttk.Button(path_frame, text="参照", command=self._choose_input).grid(row=0, column=2, pady=6, padx=(8, 0))
        ttk.Button(path_frame, text="プレビュー更新", command=self._load_preview_for_current_input).grid(
            row=0,
            column=3,
            pady=6,
            padx=(8, 0),
        )

        ttk.Label(path_frame, text="出力フォルダ").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=6)
        ttk.Entry(path_frame, textvariable=self.output_dir_var).grid(row=1, column=1, sticky="ew", pady=6)
        ttk.Button(path_frame, text="参照", command=self._choose_output).grid(row=1, column=2, pady=6, padx=(8, 0))

        settings_frame = ttk.LabelFrame(root_frame, text="書き出し設定", padding=12)
        settings_frame.pack(fill="x", pady=(12, 0))
        # 偶数列 = 見出しラベル (固定幅) / 奇数列 = 入力欄 (余白を分配)。
        # 複数のコントロールをまとめたい行は内側 Frame を使い、列構成を崩さない。
        for column in range(SETTINGS_COLUMN_COUNT):
            settings_frame.columnconfigure(column, weight=0 if column % 2 == 0 else 1)

        def settings_label(text: str, row: int, column: int) -> None:
            padx = (0, 8) if column == 0 else (16, 8)
            ttk.Label(settings_frame, text=text).grid(row=row, column=column, sticky="w", padx=padx, pady=6)

        def settings_group(row: int, column: int, columnspan: int = 1) -> ttk.Frame:
            group = ttk.Frame(settings_frame)
            group.grid(row=row, column=column, columnspan=columnspan, sticky="w", pady=6)
            return group

        settings_label("FPS", 0, 0)
        ttk.Entry(settings_frame, width=12, textvariable=self.fps_var).grid(row=0, column=1, sticky="w", pady=6)

        settings_label("画角 (h_fov)", 0, 2)
        ttk.Entry(settings_frame, width=12, textvariable=self.fov_var).grid(row=0, column=3, sticky="w", pady=6)

        settings_label("幅", 0, 4)
        ttk.Entry(settings_frame, width=12, textvariable=self.width_var).grid(row=0, column=5, sticky="w", pady=6)

        settings_label("高さ", 0, 6)
        ttk.Entry(settings_frame, width=12, textvariable=self.height_var).grid(row=0, column=7, sticky="w", pady=6)

        settings_label("並列数", 1, 0)
        ttk.Entry(settings_frame, width=12, textvariable=self.parallelism_var).grid(row=1, column=1, sticky="w", pady=6)

        settings_label("GPU", 1, 2)
        gpu_group = settings_group(1, 3)
        gpu_check = ttk.Checkbutton(
            gpu_group,
            text="CUDAデコードを使う",
            variable=self.use_gpu_var,
        )
        gpu_check.pack(side="left")
        if not self.cuda_available:
            gpu_check.state(["disabled"])
        gpu_status_text = "CUDA利用可" if self.cuda_available else "CUDA未検出"
        ttk.Label(gpu_group, text=gpu_status_text).pack(side="left", padx=(12, 0))

        settings_label("再生", 1, 4)
        ttk.Checkbutton(
            settings_frame,
            text="逆再生",
            variable=self.reverse_var,
        ).grid(row=1, column=5, sticky="w", pady=6)

        settings_label("命名", 1, 6)
        ttk.Checkbutton(
            settings_frame,
            text="奇数フレームで方向indexを逆順",
            variable=self.reverse_direction_index_on_odd_var,
        ).grid(row=1, column=7, sticky="w", pady=6)

        settings_label("実行", 2, 0)
        run_group = settings_group(2, 1)
        ttk.Checkbutton(run_group, text="画像抽出", variable=self.run_extract_var).pack(side="left")
        ttk.Checkbutton(run_group, text="マスク生成", variable=self.generate_masks_var).pack(side="left", padx=(12, 0))

        settings_label("マスク対象", 2, 2)
        category_group = settings_group(2, 3, columnspan=SETTINGS_COLUMN_COUNT - 3)
        for category_index, (category_text, category_var) in enumerate(
            (
                ("空", self.mask_sky_var),
                ("人", self.mask_person_var),
                ("車", self.mask_car_var),
                ("木", self.mask_tree_var),
            )
        ):
            ttk.Checkbutton(category_group, text=category_text, variable=category_var).pack(
                side="left",
                padx=(0, 0) if category_index == 0 else (14, 0),
            )

        settings_label("マスク", 3, 0)
        mask_group = settings_group(3, 1, columnspan=SETTINGS_COLUMN_COUNT - 1)
        ttk.Label(mask_group, text="SegFormerで対象を除外").pack(side="left")
        ttk.Label(mask_group, text=f"バッチ数 (1-{MASK_BATCH_SIZE_LIMIT})").pack(side="left", padx=(16, 6))
        ttk.Entry(mask_group, width=6, textvariable=self.mask_parallelism_var).pack(side="left")
        ttk.Label(mask_group, text="細かさ").pack(side="left", padx=(16, 6))
        ttk.Combobox(
            mask_group,
            textvariable=self.mask_detail_level_var,
            values=list(MASK_DETAIL_PRESETS.keys()),
            width=8,
            state="readonly",
        ).pack(side="left")
        ttk.Label(mask_group, text="除外しきい値").pack(side="left", padx=(16, 6))
        ttk.Entry(mask_group, width=8, textvariable=self.mask_confidence_threshold_var).pack(side="left")

        ttk.Label(
            settings_frame,
            text="mask は `元画像.jpg.mask.png` / 白=使用, 黒=除外",
        ).grid(row=4, column=0, columnspan=SETTINGS_COLUMN_COUNT, sticky="w", pady=(6, 0))

        settings_label("抽出", 5, 0)
        single_pass_group = settings_group(5, 1, columnspan=SETTINGS_COLUMN_COUNT - 1)
        ttk.Checkbutton(
            single_pass_group,
            text="単一デコードで全方向を抽出",
            variable=self.single_pass_var,
        ).pack(side="left")
        ttk.Label(
            single_pass_group,
            text="入力のデコードを方向数回から1回に減らします",
        ).pack(side="left", padx=(12, 0))
        ttk.Checkbutton(
            single_pass_group,
            text="RealityScan用XMPを書き出す",
            variable=self.write_xmp_var,
        ).pack(side="left", padx=(24, 0))
        ttk.Label(
            single_pass_group,
            text="焦点距離と歪みゼロを事前情報として渡します",
        ).pack(side="left", padx=(12, 0))

        # 実行ボタンを含む操作バーを先に下端へ固定する。
        # workspace より後に pack すると縦幅が足りない時に潰れて見えなくなる。
        action_frame = ttk.Frame(root_frame, padding=(0, 12, 0, 0))
        action_frame.pack(side="bottom", fill="x")

        ttk.Label(action_frame, textvariable=self.status_var).pack(side="left")
        ttk.Button(action_frame, text="停止", command=self._request_stop).pack(side="right")
        ttk.Button(action_frame, text="実行", command=self._start_processing).pack(side="right", padx=(0, 8))

        workspace_frame = ttk.Frame(root_frame)
        workspace_frame.pack(side="top", fill="both", expand=True, pady=(12, 0))
        workspace_frame.columnconfigure(0, weight=3)
        workspace_frame.columnconfigure(1, weight=2)
        # 余ったスペースはプレビュー側 (row 0) を優先して広げる。
        workspace_frame.rowconfigure(0, weight=4)
        workspace_frame.rowconfigure(1, weight=1)

        preview_frame = ttk.LabelFrame(workspace_frame, text="入力動画プレビュー", padding=12)
        preview_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 12), pady=(0, 12))
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)

        self.preview_media_frame = ttk.Frame(
            preview_frame,
            width=PREVIEW_SURFACE_MIN_WIDTH,
            height=PREVIEW_SURFACE_MIN_HEIGHT,
        )
        self.preview_media_frame.grid(row=0, column=0, sticky="nsew")
        # place した子でサイズが押し広げられないようにする (常に親の実サイズが正)。
        self.preview_media_frame.pack_propagate(False)
        self.preview_media_frame.grid_propagate(False)
        self.preview_media_frame.bind("<Configure>", self._on_preview_media_configure)

        self.preview_canvas = tk.Canvas(
            self.preview_media_frame,
            width=PREVIEW_SURFACE_MIN_WIDTH,
            height=PREVIEW_SURFACE_MIN_HEIGHT,
            background="#0f172a",
            highlightthickness=1,
            highlightbackground="#334155",
        )
        self.preview_canvas.place(
            x=0,
            y=0,
            width=PREVIEW_SURFACE_MIN_WIDTH,
            height=PREVIEW_SURFACE_MIN_HEIGHT,
        )

        self.preview_video_frame = tk.Frame(
            self.preview_media_frame,
            background="#000000",
            highlightthickness=0,
            bd=0,
        )

        preview_meta_frame = ttk.Frame(preview_frame, padding=(0, 10, 0, 0))
        preview_meta_frame.grid(row=1, column=0, sticky="ew")
        preview_meta_frame.columnconfigure(1, weight=1)
        preview_controls_frame = ttk.Frame(preview_meta_frame, padding=(0, 8, 0, 0))
        preview_controls_frame.grid(row=0, column=0, columnspan=2, sticky="ew")
        preview_controls_frame.columnconfigure(1, weight=1)
        preview_controls_frame.columnconfigure(3, weight=0)

        ttk.Button(
            preview_controls_frame,
            textvariable=self.preview_play_button_var,
            command=self._toggle_preview_playback,
            width=8,
        ).grid(row=0, column=0, sticky="w")
        self.preview_seek_scale = ttk.Scale(
            preview_controls_frame,
            from_=0,
            to=1,
            variable=self.preview_seek_var,
            command=self._on_preview_seek_scale_changed,
        )
        self.preview_seek_scale.grid(row=0, column=1, sticky="ew", padx=(12, 12))
        self.preview_seek_scale.bind("<ButtonPress-1>", self._on_preview_seek_press)
        self.preview_seek_scale.bind("<ButtonRelease-1>", self._on_preview_seek_release)
        ttk.Label(preview_controls_frame, textvariable=self.preview_time_var).grid(row=0, column=2, sticky="e")
        ttk.Checkbutton(
            preview_controls_frame,
            text="枠位置優先",
            variable=self.preview_overlay_sync_var,
        ).grid(row=0, column=3, sticky="e", padx=(12, 0))

        directions_frame = ttk.Frame(workspace_frame)
        directions_frame.grid(row=0, column=1, sticky="nsew", pady=(0, 12))
        directions_frame.columnconfigure(0, weight=3)
        directions_frame.columnconfigure(1, weight=2)
        directions_frame.rowconfigure(1, weight=1)

        directions_header_frame = ttk.Frame(directions_frame)
        directions_header_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        directions_header_frame.columnconfigure(1, weight=1)

        ttk.Label(directions_header_frame, text="書き出し方向").grid(row=0, column=0, sticky="w")
        self.direction_tabs_frame = ttk.Frame(directions_header_frame)
        self.direction_tabs_frame.grid(row=0, column=1, sticky="ew", padx=(12, 8))
        ttk.Button(
            directions_header_frame,
            text="＋",
            width=4,
            command=self._add_direction_set,
        ).grid(row=0, column=2, sticky="e")

        # Treeview とスクロールバーは専用フレームに入れる。
        # 同じセルに grid すると重なってテーブルの右端を覆ってしまう。
        table_frame = ttk.Frame(directions_frame)
        table_frame.grid(row=1, column=0, sticky="nsew")
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)

        self.direction_table = ttk.Treeview(
            table_frame,
            columns=("index", "yaw", "pitch"),
            show="headings",
            selectmode="browse",
            # 要求高さを小さくしておき、余ったスペースは weight で伸ばす。
            height=4,
        )
        self.direction_table.heading("index", text="index")
        self.direction_table.heading("yaw", text="yaw")
        self.direction_table.heading("pitch", text="pitch")
        self.direction_table.column("index", width=56, minwidth=44, anchor="center")
        self.direction_table.column("yaw", width=84, minwidth=60, anchor="center")
        self.direction_table.column("pitch", width=84, minwidth=60, anchor="center")
        self.direction_table.grid(row=0, column=0, sticky="nsew")
        self.direction_table.bind("<<TreeviewSelect>>", self._on_direction_selected)

        table_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.direction_table.yview)
        table_scroll.grid(row=0, column=1, sticky="ns")
        self.direction_table.configure(yscrollcommand=table_scroll.set)

        editor_frame = ttk.Frame(directions_frame, padding=(12, 0, 0, 0))
        editor_frame.grid(row=1, column=1, sticky="nsew")
        editor_frame.columnconfigure(1, weight=1)
        editor_frame.columnconfigure(3, weight=1)

        ttk.Label(editor_frame, text="yaw").grid(row=0, column=0, sticky="w", padx=(0, 6), pady=(0, 6))
        ttk.Entry(editor_frame, textvariable=self.yaw_var, width=8).grid(row=0, column=1, sticky="ew", pady=(0, 6))
        ttk.Label(editor_frame, text="pitch").grid(row=0, column=2, sticky="w", padx=(10, 6), pady=(0, 6))
        ttk.Entry(editor_frame, textvariable=self.pitch_var, width=8).grid(row=0, column=3, sticky="ew", pady=(0, 6))

        button_frame = ttk.Frame(editor_frame)
        button_frame.grid(row=1, column=0, columnspan=4, sticky="ew")
        for button_column in range(4):
            button_frame.columnconfigure(button_column, weight=1)
        for button_column, (button_text, button_command) in enumerate(
            (
                ("追加", self._add_direction),
                ("更新", self._update_selected_direction),
                ("削除", self._remove_selected_direction),
                ("全削除", self._clear_directions),
            )
        ):
            ttk.Button(button_frame, text=button_text, width=6, command=button_command).grid(
                row=0,
                column=button_column,
                sticky="ew",
                padx=(0 if button_column == 0 else 4, 0),
            )

        preset_frame = ttk.LabelFrame(editor_frame, text="水平リング生成", padding=8)
        preset_frame.grid(row=2, column=0, columnspan=4, sticky="ew", pady=(10, 0))
        preset_frame.columnconfigure(1, weight=1)
        preset_frame.columnconfigure(3, weight=1)

        ttk.Label(preset_frame, text="方向数").grid(row=0, column=0, sticky="w", padx=(0, 6))
        ttk.Entry(preset_frame, textvariable=self.ring_count_var, width=6).grid(row=0, column=1, sticky="ew")
        ttk.Label(preset_frame, text="pitch").grid(row=0, column=2, sticky="w", padx=(10, 6))
        ttk.Entry(preset_frame, textvariable=self.ring_pitch_var, width=6).grid(row=0, column=3, sticky="ew")

        ttk.Button(preset_frame, text="生成して置換", command=self._replace_with_ring).grid(
            row=1,
            column=0,
            columnspan=4,
            sticky="ew",
            pady=(8, 0),
        )

        preview_status_frame = ttk.Frame(editor_frame, padding=(0, 10, 0, 0))
        preview_status_frame.grid(row=3, column=0, columnspan=4, sticky="nsew")
        preview_status_frame.columnconfigure(0, weight=1)
        editor_frame.rowconfigure(3, weight=1)

        self.preview_info_label = ttk.Label(preview_status_frame, textvariable=self.preview_info_var, justify="left")
        self.preview_info_label.grid(row=0, column=0, sticky="w")
        self.preview_selection_label = ttk.Label(
            preview_status_frame,
            textvariable=self.preview_selection_var,
            justify="left",
        )
        self.preview_selection_label.grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.preview_legend_label = ttk.Label(
            preview_status_frame,
            text="色ごとの枠と点が書き出し位置です。白い外枠が選択中の方向です。",
            justify="left",
        )
        self.preview_legend_label.grid(row=2, column=0, sticky="w", pady=(4, 0))
        # 折り返し幅は実際の列幅に追従させる (固定値だと狭い時に切れる)。
        preview_status_frame.bind("<Configure>", self._on_preview_status_configure)

        log_frame = ttk.LabelFrame(workspace_frame, text="ログ", padding=12)
        log_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 12))
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)

        self.log_text = tk.Text(log_frame, height=4, width=40, wrap="word", state="disabled")
        self.log_text.grid(row=0, column=0, sticky="nsew")

        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        log_scroll.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=log_scroll.set)

        sphere_frame = ttk.LabelFrame(workspace_frame, text="方向3Dビュー", padding=10)
        sphere_frame.grid(row=1, column=1, sticky="nsew")
        sphere_frame.columnconfigure(0, weight=1)
        sphere_frame.rowconfigure(0, weight=1)

        self.sphere_canvas = tk.Canvas(
            sphere_frame,
            width=self.sphere_canvas_size,
            height=self.sphere_canvas_size,
            background="#0b1220",
            highlightthickness=1,
            highlightbackground="#334155",
        )
        self.sphere_canvas.grid(row=0, column=0, sticky="nsew")
        self.sphere_canvas.bind("<ButtonPress-1>", self._on_sphere_drag_start)
        self.sphere_canvas.bind("<B1-Motion>", self._on_sphere_drag)
        self.sphere_canvas.bind("<Configure>", self._on_sphere_canvas_configure)

        self.sphere_hint_label = ttk.Label(
            sphere_frame,
            text="ドラッグで回転。手前ほど明るく、視野枠も表示します。",
            justify="left",
        )
        self.sphere_hint_label.grid(row=1, column=0, sticky="w", pady=(8, 0))
        sphere_frame.bind("<Configure>", self._on_sphere_frame_configure)

        self._refresh_direction_tabs()
        self._render_preview_overlay()

    def _on_preview_status_configure(self, event: tk.Event[tk.Misc]) -> None:
        wrap_length = max(120, int(event.width) - 4)
        for label in (self.preview_info_label, self.preview_selection_label, self.preview_legend_label):
            label.configure(wraplength=wrap_length)

    def _on_sphere_frame_configure(self, event: tk.Event[tk.Misc]) -> None:
        self.sphere_hint_label.configure(wraplength=max(120, int(event.width) - 24))

    def _bind_preview_refresh(self) -> None:
        for variable in (self.fov_var, self.width_var, self.height_var):
            variable.trace_add("write", self._on_preview_settings_changed)
        self.preview_overlay_sync_var.trace_add("write", self._on_preview_mode_changed)

    def _sync_active_direction_set(self) -> None:
        """表示中の方向リストをアクティブな DirectionSet に確実に反映させる。"""
        if 0 <= self.active_direction_set_index < len(self.direction_sets):
            self.direction_sets[self.active_direction_set_index].directions = self.directions

    def _build_settings_payload(self) -> dict[str, object]:
        self._sync_active_direction_set()
        return {
            "input_path": self.input_path_var.get().strip(),
            "output_dir": self.output_dir_var.get().strip(),
            "fps": self.fps_var.get().strip(),
            "fov": self.fov_var.get().strip(),
            "width": self.width_var.get().strip(),
            "height": self.height_var.get().strip(),
            "parallelism": self.parallelism_var.get().strip(),
            "mask_parallelism": self.mask_parallelism_var.get().strip(),
            "run_extract": bool(self.run_extract_var.get()),
            "use_gpu": bool(self.use_gpu_var.get()),
            "reverse": bool(self.reverse_var.get()),
            "single_pass": bool(self.single_pass_var.get()),
            "write_xmp": bool(self.write_xmp_var.get()),
            "reverse_direction_index_on_odd": bool(self.reverse_direction_index_on_odd_var.get()),
            "generate_masks": bool(self.generate_masks_var.get()),
            "mask_detail_level": self.mask_detail_level_var.get().strip(),
            "mask_confidence_threshold": self.mask_confidence_threshold_var.get().strip(),
            "mask_sky": bool(self.mask_sky_var.get()),
            "mask_person": bool(self.mask_person_var.get()),
            "mask_car": bool(self.mask_car_var.get()),
            "mask_tree": bool(self.mask_tree_var.get()),
            "preview_overlay_sync": bool(self.preview_overlay_sync_var.get()),
            "yaw": self.yaw_var.get().strip(),
            "pitch": self.pitch_var.get().strip(),
            "ring_count": self.ring_count_var.get().strip(),
            "ring_pitch": self.ring_pitch_var.get().strip(),
            "active_direction_set_index": self.active_direction_set_index,
            "direction_sets": [
                {
                    "name": direction_set.name,
                    "directions": [
                        {
                            "yaw": direction.yaw,
                            "pitch": direction.pitch,
                        }
                        for direction in direction_set.directions
                    ],
                }
                for direction_set in self.direction_sets
            ],
            "directions": [
                {
                    "yaw": direction.yaw,
                    "pitch": direction.pitch,
                }
                for direction in self.directions
            ],
        }

    def _save_persisted_settings(self) -> None:
        payload = self._build_settings_payload()
        self.settings_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _load_persisted_settings(self) -> None:
        self.suspend_preview_mode_reload = True
        if not self.settings_path.is_file():
            self.suspend_preview_mode_reload = False
            return

        try:
            payload = json.loads(self.settings_path.read_text(encoding="utf-8"))
        except Exception:
            self.suspend_preview_mode_reload = False
            return

        if not isinstance(payload, dict):
            self.suspend_preview_mode_reload = False
            return

        string_keys = (
            ("input_path", self.input_path_var),
            ("output_dir", self.output_dir_var),
            ("fps", self.fps_var),
            ("fov", self.fov_var),
            ("width", self.width_var),
            ("height", self.height_var),
            ("parallelism", self.parallelism_var),
            ("mask_parallelism", self.mask_parallelism_var),
            ("yaw", self.yaw_var),
            ("pitch", self.pitch_var),
            ("ring_count", self.ring_count_var),
            ("ring_pitch", self.ring_pitch_var),
        )
        for key, variable in string_keys:
            value = payload.get(key)
            if isinstance(value, str):
                variable.set(value)

        saved_use_gpu = payload.get("use_gpu")
        if isinstance(saved_use_gpu, bool):
            self.use_gpu_var.set(saved_use_gpu and self.cuda_available)

        saved_run_extract = payload.get("run_extract")
        if isinstance(saved_run_extract, bool):
            self.run_extract_var.set(saved_run_extract)

        saved_single_pass = payload.get("single_pass")
        if isinstance(saved_single_pass, bool):
            self.single_pass_var.set(saved_single_pass)

        saved_write_xmp = payload.get("write_xmp")
        if isinstance(saved_write_xmp, bool):
            self.write_xmp_var.set(saved_write_xmp)

        saved_reverse = payload.get("reverse")
        if isinstance(saved_reverse, bool):
            self.reverse_var.set(saved_reverse)

        saved_reverse_direction_index_on_odd = payload.get("reverse_direction_index_on_odd")
        if isinstance(saved_reverse_direction_index_on_odd, bool):
            self.reverse_direction_index_on_odd_var.set(saved_reverse_direction_index_on_odd)

        saved_generate_masks = payload.get("generate_masks")
        if isinstance(saved_generate_masks, bool):
            self.generate_masks_var.set(saved_generate_masks)

        saved_mask_detail_level = payload.get("mask_detail_level")
        if isinstance(saved_mask_detail_level, str) and saved_mask_detail_level in MASK_DETAIL_PRESETS:
            self.mask_detail_level_var.set(saved_mask_detail_level)
        else:
            saved_mask_detail = payload.get("mask_detail")
            if isinstance(saved_mask_detail, bool):
                self.mask_detail_level_var.set("高" if saved_mask_detail else DEFAULT_MASK_DETAIL_LEVEL)

        saved_mask_confidence_threshold = payload.get("mask_confidence_threshold")
        if isinstance(saved_mask_confidence_threshold, str):
            self.mask_confidence_threshold_var.set(saved_mask_confidence_threshold)

        for key, variable in (
            ("mask_sky", self.mask_sky_var),
            ("mask_person", self.mask_person_var),
            ("mask_car", self.mask_car_var),
            ("mask_tree", self.mask_tree_var),
            ("preview_overlay_sync", self.preview_overlay_sync_var),
        ):
            saved_value = payload.get(key)
            if isinstance(saved_value, bool):
                variable.set(saved_value)

        parsed_direction_sets: list[DirectionSet] = []
        saved_direction_sets = payload.get("direction_sets")
        if isinstance(saved_direction_sets, list):
            for set_index, raw_set in enumerate(saved_direction_sets, start=1):
                if not isinstance(raw_set, dict):
                    continue
                raw_name = raw_set.get("name")
                set_name = raw_name if isinstance(raw_name, str) and raw_name.strip() else f"セット{set_index}"
                parsed_directions = self._parse_direction_list(raw_set.get("directions"))
                if parsed_directions:
                    parsed_direction_sets.append(DirectionSet(name=set_name, directions=parsed_directions))

        if not parsed_direction_sets:
            parsed_directions = self._parse_direction_list(payload.get("directions"))
            if parsed_directions:
                parsed_direction_sets = [DirectionSet(name="セット1", directions=parsed_directions)]

        if parsed_direction_sets:
            self.direction_sets = parsed_direction_sets

        saved_active_direction_set_index = payload.get("active_direction_set_index")
        if isinstance(saved_active_direction_set_index, int):
            self.active_direction_set_index = max(0, min(saved_active_direction_set_index, len(self.direction_sets) - 1))
        else:
            self.active_direction_set_index = 0

        self.directions = self.direction_sets[self.active_direction_set_index].directions
        self.suspend_preview_mode_reload = False

    def _parse_direction_list(self, raw_directions: object) -> list[Direction]:
        if not isinstance(raw_directions, list):
            return []

        parsed_directions: list[Direction] = []
        for item in raw_directions:
            if not isinstance(item, dict):
                continue
            try:
                yaw = normalize_angle(float(item["yaw"]))
                pitch = normalize_angle(float(item["pitch"]))
            except (KeyError, TypeError, ValueError):
                continue
            parsed_directions.append(Direction(yaw=yaw, pitch=pitch))
        return parsed_directions

    def _refresh_direction_tabs(self) -> None:
        for widget in self.direction_tabs_frame.winfo_children():
            widget.destroy()

        for index, direction_set in enumerate(self.direction_sets):
            is_active = index == self.active_direction_set_index
            button = tk.Button(
                self.direction_tabs_frame,
                text=direction_set.name,
                relief=tk.SUNKEN if is_active else tk.RAISED,
                bd=1,
                padx=10,
                pady=2,
                state=tk.DISABLED if is_active else tk.NORMAL,
                command=lambda idx=index: self._switch_direction_set(idx),
            )
            button.pack(side="left", padx=(0, 6))

        # タブが増えて必要幅が伸びたら最小ウィンドウサイズも追従させる。
        if getattr(self, "log_drain_after_id", None) is not None:
            self._apply_window_size_limits()

    def _switch_direction_set(self, index: int) -> None:
        if not 0 <= index < len(self.direction_sets):
            return
        self._sync_active_direction_set()
        self.active_direction_set_index = index
        self.directions = self.direction_sets[index].directions
        self._refresh_direction_tabs()
        self._refresh_direction_table(select_index=0)
        self.status_var.set(f"{self.direction_sets[index].name} に切り替えました。")

    def _add_direction_set(self) -> None:
        self._sync_active_direction_set()
        next_index = len(self.direction_sets) + 1
        existing_names = {direction_set.name for direction_set in self.direction_sets}
        candidate_name = f"セット{next_index}"
        while candidate_name in existing_names:
            next_index += 1
            candidate_name = f"セット{next_index}"

        self.direction_sets.append(DirectionSet(name=candidate_name, directions=[Direction(yaw=0.0, pitch=0.0)]))
        self.active_direction_set_index = len(self.direction_sets) - 1
        self.directions = self.direction_sets[self.active_direction_set_index].directions
        self._refresh_direction_tabs()
        self._refresh_direction_table(select_index=0)
        self.status_var.set(f"{candidate_name} を追加しました。")

    def _on_preview_settings_changed(self, *_args: object) -> None:
        self._render_preview_overlay()
        self._render_direction_sphere()

    def _on_preview_mode_changed(self, *_args: object) -> None:
        if self.suspend_preview_mode_reload:
            return
        if self.input_path_var.get().strip():
            self._load_preview_for_current_input(show_errors=False)

    def _get_segformer_mask_generator(self) -> SegFormerMaskGenerator:
        if self.segformer_mask_generator is None:
            self.segformer_mask_generator = SegFormerMaskGenerator()
        return self.segformer_mask_generator

    def _selected_mask_categories(self) -> list[str]:
        selected_categories: list[str] = []
        if self.mask_sky_var.get():
            selected_categories.append("sky")
        if self.mask_person_var.get():
            selected_categories.append("person")
        if self.mask_car_var.get():
            selected_categories.append("car")
        if self.mask_tree_var.get():
            selected_categories.append("tree")
        return selected_categories

    def _build_mask_image_jobs(self, output_dir: Path, video_stem: str | None) -> list[ExtractedImageJob]:
        if not output_dir.is_dir():
            return []
        image_paths = sorted(output_dir.glob(build_extracted_image_glob(video_stem)))
        return [
            ExtractedImageJob(
                image_path=image_path,
                mask_path=build_mask_output_path(image_path),
            )
            for image_path in image_paths
            if image_path.is_file()
        ]

    def _close_preview_capture(self) -> None:
        self._cancel_preview_vlc_poll()
        self._cancel_preview_snapshot()
        if self.preview_capture is not None:
            self.preview_capture.release()
            self.preview_capture = None
        if self.preview_proxy_path is not None:
            self.preview_proxy_path.unlink(missing_ok=True)
            self.preview_proxy_path = None
        self._stop_preview_vlc()
        # VLC を使うかどうかはモードで判定する。プレイヤーの生存で判定すると
        # 一度高速再生を使った後に「枠位置優先」へ戻せなくなる。
        self.preview_mode = "none"
        self.preview_current_video_path = None
        self.preview_last_frame_bgr = None
        self.preview_total_frames = 0
        self.preview_current_frame_index = 0
        self.preview_fps = 0.0
        self.preview_duration_seconds = 0.0

    def _cancel_preview_playback(self) -> None:
        if self.preview_playback_after_id is not None:
            self.root.after_cancel(self.preview_playback_after_id)
            self.preview_playback_after_id = None

    def _pause_preview_playback(self) -> None:
        self.preview_is_playing = False
        self.preview_play_button_var.set("再生")
        self._cancel_preview_playback()
        self._cancel_preview_vlc_poll()

    def _cancel_preview_seek(self) -> None:
        if self.preview_seek_after_id is not None:
            self.root.after_cancel(self.preview_seek_after_id)
            self.preview_seek_after_id = None

    def _reset_preview_controls(self) -> None:
        self.preview_slider_internal_update = True
        self.preview_seek_var.set(0.0)
        self.preview_seek_scale.configure(from_=0, to=1)
        self.preview_slider_internal_update = False
        self.preview_time_var.set("00:00 / 00:00")
        self.preview_play_button_var.set("再生")
        self._cancel_preview_seek()

    def _preview_uses_vlc(self) -> bool:
        return self.preview_mode == "vlc" and self.preview_vlc_player is not None

    def _stop_preview_vlc(self) -> None:
        if self.preview_vlc_player is None:
            return
        try:
            self.preview_vlc_player.stop()
        except Exception:
            pass

    def _release_preview_vlc(self) -> None:
        self._cancel_preview_vlc_poll()
        self._cancel_preview_snapshot()
        self._stop_preview_vlc()
        for attribute in ("preview_vlc_player", "preview_vlc_instance"):
            handle = getattr(self, attribute)
            if handle is None:
                continue
            try:
                handle.release()
            except Exception:
                pass
            setattr(self, attribute, None)
        self.preview_vlc_module = None
        self.preview_mode = "none"

    def _cancel_preview_vlc_poll(self) -> None:
        if self.preview_vlc_poll_after_id is not None:
            self.root.after_cancel(self.preview_vlc_poll_after_id)
            self.preview_vlc_poll_after_id = None

    def _cancel_preview_snapshot(self) -> None:
        if self.preview_snapshot_after_id is not None:
            self.root.after_cancel(self.preview_snapshot_after_id)
            self.preview_snapshot_after_id = None

    def _preview_surface_bounds(self) -> tuple[int, int]:
        """プレビューを描ける実領域。ウィジェットが未実体化の間は最小値を返す。"""
        available_width = self.preview_media_frame.winfo_width()
        available_height = self.preview_media_frame.winfo_height()
        if available_width <= 1 or available_height <= 1:
            return PREVIEW_SURFACE_MIN_WIDTH, PREVIEW_SURFACE_MIN_HEIGHT
        return max(1, available_width), max(1, available_height)

    def _show_preview_video_widget(self, show_video: bool) -> None:
        # Canvas と VLC 描画用 Frame は必ず同じ矩形に置く。位置がずれると
        # 再生 / 一時停止のたびに映像が飛ぶ。
        place_options = {
            "x": self.preview_canvas_offset_x,
            "y": self.preview_canvas_offset_y,
            "width": max(1, self.preview_canvas_width),
            "height": max(1, self.preview_canvas_height),
        }
        if show_video:
            self.preview_canvas.place_forget()
            self.preview_video_frame.place(**place_options)
        else:
            self.preview_video_frame.place_forget()
            self.preview_canvas.place(**place_options)

    def _ensure_preview_vlc(self) -> bool:
        if self.preview_vlc_player is not None:
            return True

        try:
            vlc = get_vlc()
            self.preview_vlc_module = vlc
            self.preview_vlc_instance = vlc.Instance("--quiet", "--no-video-title-show")
            self.root.update_idletasks()
            self.preview_vlc_player = self.preview_vlc_instance.media_player_new()
            self.preview_vlc_player.set_hwnd(self.preview_video_frame.winfo_id())
            return True
        except Exception as error:
            self._append_log(f"ERROR: VLC 初期化に失敗しました: {error}")
            self.preview_vlc_module = None
            self.preview_vlc_instance = None
            self.preview_vlc_player = None
            return False

    def _set_preview_vlc_time_from_frame(self, frame_index: int) -> None:
        if self.preview_vlc_player is None:
            return
        if self.preview_fps <= 0.0:
            return
        time_ms = int(round((frame_index / self.preview_fps) * 1000.0))
        self.preview_vlc_player.set_time(max(0, time_ms))

    def _schedule_preview_snapshot(self, delay_ms: int = 180) -> None:
        if not self._preview_uses_vlc():
            return

        self._cancel_preview_snapshot()

        def take_snapshot() -> None:
            self.preview_snapshot_after_id = None
            self._capture_preview_snapshot()

        self.preview_snapshot_after_id = self.root.after(delay_ms, take_snapshot)

    def _capture_preview_snapshot(self) -> None:
        if self.preview_vlc_player is None:
            return

        with tempfile.NamedTemporaryFile(prefix="insta360_preview_snapshot_", suffix=".png", delete=False) as temporary:
            snapshot_path = Path(temporary.name)

        try:
            snapshot_width = max(MIN_SNAPSHOT_SIZE, self.preview_canvas_width)
            snapshot_height = max(MIN_SNAPSHOT_SIZE, self.preview_canvas_height)
            try:
                result = self.preview_vlc_player.video_take_snapshot(
                    0,
                    str(snapshot_path),
                    snapshot_width,
                    snapshot_height,
                )
            except Exception as error:
                self._append_log(f"ERROR: プレビュー静止画の取得に失敗しました: {error}")
                return
            if result != 0 or not snapshot_path.is_file():
                return

            try:
                with Image.open(snapshot_path) as pil_image:
                    self.preview_photo = ImageTk.PhotoImage(pil_image.convert("RGB"))
                self.preview_photo_from_snapshot = True
            except Exception as error:
                self._append_log(f"ERROR: プレビュー静止画の読み込みに失敗しました: {error}")
                return
            self._show_preview_video_widget(False)
            self._render_preview_overlay()
        finally:
            snapshot_path.unlink(missing_ok=True)

    def _poll_preview_vlc(self) -> None:
        self._cancel_preview_vlc_poll()
        if self.preview_vlc_player is None or not self.preview_is_playing:
            return

        current_time_ms = max(0, int(self.preview_vlc_player.get_time() or 0))
        if self.preview_fps > 0.0:
            self.preview_current_frame_index = max(0, min(int(round((current_time_ms / 1000.0) * self.preview_fps)), max(self.preview_total_frames - 1, 0)))

        self.preview_slider_internal_update = True
        self.preview_seek_var.set(float(self.preview_current_frame_index))
        self.preview_slider_internal_update = False
        self._set_preview_time_for_frame_index(self.preview_current_frame_index)

        state = self.preview_vlc_player.get_state()
        if state in (self.preview_vlc_module.State.Ended, self.preview_vlc_module.State.Stopped):
            self._pause_preview_playback()
            self._capture_preview_snapshot()
            return

        self.preview_vlc_poll_after_id = self.root.after(60, self._poll_preview_vlc)

    def _open_preview_vlc_media(self, video_path: Path) -> bool:
        if not self._ensure_preview_vlc():
            return False
        assert self.preview_vlc_instance is not None
        assert self.preview_vlc_player is not None

        media = self.preview_vlc_instance.media_new(str(video_path))
        self.preview_vlc_player.set_media(media)
        self.preview_current_video_path = video_path
        return True

    def _set_preview_preparing_state(self, preparing: bool) -> None:
        self.preview_proxy_preparing = preparing
        self.preview_seek_scale.state(["disabled"] if preparing else ["!disabled"])
        if preparing:
            self.preview_play_button_var.set("準備中")
        else:
            self.preview_play_button_var.set("再生")

    def _open_preview_capture(
        self,
        proxy_path: Path,
        video_path: Path,
        metadata: VideoMetadata,
    ) -> None:
        cv2 = get_cv2()
        self._pause_preview_playback()
        self._close_preview_capture()
        capture = cv2.VideoCapture(str(proxy_path))
        if not capture.isOpened():
            proxy_path.unlink(missing_ok=True)
            raise RuntimeError("動画プレビュー用の VideoCapture を開けませんでした。")

        raw_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if not math.isfinite(raw_fps) or raw_fps <= 0.0:
            raw_fps = 30.0

        raw_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if raw_frame_count <= 0 and metadata.duration_seconds is not None:
            raw_frame_count = max(1, int(round(metadata.duration_seconds * raw_fps)))
        if raw_frame_count <= 0:
            raw_frame_count = 1

        if metadata.duration_seconds is not None:
            duration_seconds = metadata.duration_seconds
        else:
            duration_seconds = raw_frame_count / raw_fps

        self.preview_capture = capture
        self.preview_mode = "proxy"
        self.preview_proxy_path = proxy_path
        self.preview_current_video_path = video_path
        self.preview_fps = raw_fps
        self.preview_total_frames = raw_frame_count
        self.preview_duration_seconds = duration_seconds
        self.preview_current_frame_index = 0

        self.preview_slider_internal_update = True
        self.preview_seek_scale.configure(from_=0, to=max(raw_frame_count - 1, 1))
        self.preview_seek_var.set(0.0)
        self.preview_slider_internal_update = False
        self.preview_time_var.set(f"00:00 / {format_timestamp(duration_seconds)}")

    def _open_preview_player_fast(
        self,
        video_path: Path,
        metadata: VideoMetadata,
        preview_width: int,
        preview_height: int,
    ) -> bool:
        if not self._open_preview_vlc_media(video_path):
            return False

        self.preview_capture = None
        self.preview_proxy_path = None
        self.preview_metadata = metadata
        self.preview_mode = "vlc"
        self.preview_current_video_path = video_path
        self.preview_current_frame_index = 0
        self.preview_duration_seconds = metadata.duration_seconds or 0.0
        self.preview_fps = 30.0
        self.preview_total_frames = max(1, int(round(self.preview_duration_seconds * self.preview_fps)))

        self._auto_resize_window_for_preview(preview_width, preview_height)
        self._layout_preview_surface()

        self.preview_slider_internal_update = True
        self.preview_seek_scale.configure(from_=0, to=max(self.preview_total_frames - 1, 1))
        self.preview_seek_var.set(0.0)
        self.preview_slider_internal_update = False
        self._set_preview_time_for_frame_index(0)
        self._set_preview_preparing_state(False)
        return True

    def _read_initial_preview_frame(
        self,
        video_path: Path,
        ffmpeg_path: str,
        metadata: VideoMetadata,
    ) -> object | None:
        frame = self._read_initial_frame_with_cv2(video_path)
        if frame is not None:
            return frame
        # .insv など OpenCV が開けない形式でもプレビューを出せるよう ffmpeg に退避する。
        return self._read_initial_frame_with_ffmpeg(video_path, ffmpeg_path, metadata)

    def _read_initial_frame_with_cv2(self, video_path: Path) -> object | None:
        try:
            cv2 = get_cv2()
        except Exception as error:
            self._append_log(f"ERROR: OpenCV を読み込めませんでした: {error}")
            return None

        capture = None
        try:
            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                return None
            success, frame = capture.read()
            if not success or frame is None:
                return None
            return frame
        except Exception:
            return None
        finally:
            if capture is not None:
                capture.release()

    def _read_initial_frame_with_ffmpeg(
        self,
        video_path: Path,
        ffmpeg_path: str,
        metadata: VideoMetadata,
    ) -> object | None:
        import numpy as np

        target_width, target_height = fit_size_within_bounds(
            metadata.width,
            metadata.height,
            *self._preview_surface_bounds(),
            allow_upscale=False,
        )
        preview_path: Path | None = None
        try:
            preview_path = extract_preview_image(
                video_path,
                ffmpeg_path,
                metadata,
                target_width,
                target_height,
            )
            with Image.open(preview_path) as pil_image:
                rgb_array = np.asarray(pil_image.convert("RGB"))
            # _display_preview_frame は BGR を前提にしているため並べ替える。
            return rgb_array[:, :, ::-1].copy()
        except Exception as error:
            self._append_log(f"ERROR: ffmpeg でのサムネイル取得に失敗しました: {error}")
            return None
        finally:
            if preview_path is not None:
                preview_path.unlink(missing_ok=True)

    def _prepare_preview_proxy_async(
        self,
        request_id: int,
        video_path: Path,
        ffmpeg_path: str,
        metadata: VideoMetadata,
        preview_width: int,
        preview_height: int,
    ) -> None:
        try:
            proxy_path = extract_preview_proxy_video(
                video_path,
                ffmpeg_path,
                preview_width,
                preview_height,
            )
            self.log_queue.put(
                (
                    "preview_proxy_ready",
                    {
                        "request_id": request_id,
                        "video_path": video_path,
                        "proxy_path": proxy_path,
                        "metadata": metadata,
                    },
                )
            )
        except Exception as error:
            self.log_queue.put(
                (
                    "preview_proxy_error",
                    {
                        "request_id": request_id,
                        "error": str(error),
                    },
                )
            )

    def _read_preview_frame(self, frame_index: int, sequential: bool = False) -> object | None:
        if self.preview_capture is None:
            return None

        cv2 = get_cv2()
        clamped_index = max(0, min(frame_index, max(self.preview_total_frames - 1, 0)))
        if not sequential or clamped_index != self.preview_current_frame_index + 1:
            self.preview_capture.set(cv2.CAP_PROP_POS_FRAMES, clamped_index)

        success, frame = self.preview_capture.read()
        if not success or frame is None:
            return None

        self.preview_current_frame_index = clamped_index
        return frame

    def _set_preview_time_for_current_frame(self) -> None:
        self._set_preview_time_for_frame_index(self.preview_current_frame_index)

    def _set_preview_time_for_frame_index(self, frame_index: int) -> None:
        if self.preview_fps > 0.0:
            current_seconds = max(0, frame_index) / self.preview_fps
        else:
            current_seconds = 0.0
        self.preview_time_var.set(
            f"{format_timestamp(current_seconds)} / {format_timestamp(self.preview_duration_seconds)}"
        )

    def _display_preview_frame(self, frame_bgr: object) -> None:
        self.preview_last_frame_bgr = frame_bgr
        self._update_preview_photo(frame_bgr)

        self.preview_slider_internal_update = True
        self.preview_seek_var.set(float(self.preview_current_frame_index))
        self.preview_slider_internal_update = False
        self._set_preview_time_for_current_frame()
        self._render_preview_overlay()

    def _update_preview_photo(self, frame_bgr: object) -> None:
        """BGR フレームを現在の Canvas サイズに合わせて PhotoImage 化する。"""
        self.preview_photo_from_snapshot = False
        target_width = max(1, self.preview_canvas_width)
        target_height = max(1, self.preview_canvas_height)

        try:
            cv2 = get_cv2()
            frame_array = frame_bgr
            frame_shape = getattr(frame_array, "shape", None)
            if frame_shape is not None:
                frame_height, frame_width = frame_shape[:2]
                if (frame_width, frame_height) != (target_width, target_height):
                    frame_array = cv2.resize(
                        frame_array,
                        (target_width, target_height),
                        interpolation=cv2.INTER_LINEAR,
                    )
            frame_rgb = cv2.cvtColor(frame_array, cv2.COLOR_BGR2RGB)
            self.preview_photo = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
        except Exception as error:
            # 極小サイズや壊れたフレームでもプレビュー全体を落とさない。
            self.preview_photo = None
            self._append_log(f"ERROR: プレビュー描画に失敗しました: {error}")

    def _render_preview_frame(self, frame_index: int, sequential: bool = False) -> bool:
        frame = self._read_preview_frame(frame_index, sequential=sequential)
        if frame is None:
            return False
        self._display_preview_frame(frame)
        return True

    def _request_preview_seek_render(self, frame_index: int, delay_ms: int) -> None:
        self._cancel_preview_seek()

        def apply_seek() -> None:
            self.preview_seek_after_id = None
            self._render_preview_frame(frame_index, sequential=False)

        self.preview_seek_after_id = self.root.after(delay_ms, apply_seek)

    def _schedule_next_preview_frame(self) -> None:
        if self._preview_uses_vlc():
            self._poll_preview_vlc()
            return

        self._cancel_preview_playback()
        if not self.preview_is_playing:
            return

        next_frame_index = self.preview_current_frame_index + 1
        if next_frame_index >= self.preview_total_frames:
            self._pause_preview_playback()
            return

        if not self._render_preview_frame(next_frame_index, sequential=True):
            self._pause_preview_playback()
            return

        frame_delay_ms = max(15, int(round(1000.0 / max(self.preview_fps, 1.0))))
        self.preview_playback_after_id = self.root.after(frame_delay_ms, self._schedule_next_preview_frame)

    def _toggle_preview_playback(self) -> None:
        if self._preview_uses_vlc():
            if self.preview_is_playing:
                try:
                    self.preview_vlc_player.pause()
                except Exception:
                    pass
                self._pause_preview_playback()
                self._schedule_preview_snapshot(140)
                return

            if self.preview_current_video_path is None:
                if self.input_path_var.get().strip():
                    self._load_preview_for_current_input()
                return

            self._show_preview_video_widget(True)
            try:
                self.preview_vlc_player.play()
                self._set_preview_vlc_time_from_frame(self.preview_current_frame_index)
            except Exception:
                self._show_preview_video_widget(False)
                return
            self.preview_is_playing = True
            self.preview_play_button_var.set("停止")
            self._poll_preview_vlc()
            return

        if self.preview_capture is None:
            if self.preview_proxy_preparing:
                self.status_var.set("軽量プレビューを準備中です...")
                return
            if self.input_path_var.get().strip():
                self._load_preview_for_current_input()
            return

        if self.preview_is_playing:
            self._pause_preview_playback()
            return

        if self.preview_current_frame_index >= max(self.preview_total_frames - 1, 0):
            self._render_preview_frame(0, sequential=False)
        self.preview_is_playing = True
        self.preview_play_button_var.set("停止")
        self._schedule_next_preview_frame()

    def _on_preview_seek_press(self, _event: tk.Event[tk.Misc]) -> None:
        self.preview_slider_active = True
        self._pause_preview_playback()
        if self._preview_uses_vlc():
            try:
                self.preview_vlc_player.pause()
            except Exception:
                pass
        self._cancel_preview_seek()

    def _on_preview_seek_scale_changed(self, value: str) -> None:
        if self.preview_slider_internal_update:
            return
        if self._preview_uses_vlc():
            frame_index = int(round(float(value)))
            self.preview_current_frame_index = max(0, min(frame_index, max(self.preview_total_frames - 1, 0)))
            self._set_preview_time_for_frame_index(self.preview_current_frame_index)
            self._show_preview_video_widget(True)
            self._set_preview_vlc_time_from_frame(self.preview_current_frame_index)
            if not self.preview_slider_active:
                self._schedule_preview_snapshot(100)
            return
        if self.preview_capture is None:
            if self.preview_proxy_preparing:
                if self.preview_fps > 0.0:
                    self._set_preview_time_for_frame_index(int(round(float(value))))
            return
        frame_index = int(round(float(value)))
        self._set_preview_time_for_frame_index(frame_index)
        self._request_preview_seek_render(frame_index, 120 if self.preview_slider_active else 40)

    def _on_preview_seek_release(self, _event: tk.Event[tk.Misc]) -> None:
        self.preview_slider_active = False
        if self._preview_uses_vlc():
            self._schedule_preview_snapshot(80)
            return
        if self.preview_capture is None:
            return
        frame_index = int(round(float(self.preview_seek_var.get())))
        self._cancel_preview_seek()
        self._render_preview_frame(frame_index, sequential=False)

    def _on_sphere_canvas_configure(self, _event: tk.Event[tk.Misc]) -> None:
        # 描画側が winfo_width/height を直接読むので、ここでは再描画のみ行う。
        self._render_direction_sphere()

    def _on_sphere_drag_start(self, event: tk.Event[tk.Misc]) -> None:
        self.sphere_drag_last = (int(event.x), int(event.y))

    def _on_sphere_drag(self, event: tk.Event[tk.Misc]) -> None:
        if self.sphere_drag_last is None:
            self.sphere_drag_last = (int(event.x), int(event.y))
            return

        last_x, last_y = self.sphere_drag_last
        delta_x = int(event.x) - last_x
        delta_y = int(event.y) - last_y
        self.sphere_view_yaw = normalize_angle(self.sphere_view_yaw + (delta_x * 0.8))
        self.sphere_view_pitch = max(-89.0, min(89.0, self.sphere_view_pitch - (delta_y * 0.8)))
        self.sphere_drag_last = (int(event.x), int(event.y))
        self._render_direction_sphere()

    def _project_sphere_vector(
        self,
        vector: tuple[float, float, float],
        radius: float,
        canvas_center_x: float,
        canvas_center_y: float,
    ) -> tuple[float, float, float]:
        rotated_vector = apply_view_rotation(vector, self.sphere_view_yaw, self.sphere_view_pitch)
        vector_x, vector_y, vector_z = rotated_vector
        return (
            canvas_center_x + (vector_x * radius),
            canvas_center_y - (vector_y * radius),
            vector_z,
        )

    def _build_direction_plane_vertices(
        self,
        direction: Direction,
        horizontal_fov: float,
        aspect_ratio: float,
    ) -> tuple[tuple[float, float, float], list[tuple[float, float, float]]]:
        clamped_fov = max(1.0, min(horizontal_fov, 179.0))
        tangent_horizontal = math.tan(math.radians(clamped_fov) / 2.0)
        tangent_vertical = tangent_horizontal / max(aspect_ratio, 1e-6)
        plane_distance = 1.15
        plane_half_width = tangent_horizontal * plane_distance
        plane_half_height = tangent_vertical * plane_distance

        center_point = rotate_local_point((0.0, 0.0, plane_distance), direction.yaw, direction.pitch)
        corner_points = [
            rotate_local_point((-plane_half_width, plane_half_height, plane_distance), direction.yaw, direction.pitch),
            rotate_local_point((plane_half_width, plane_half_height, plane_distance), direction.yaw, direction.pitch),
            rotate_local_point((plane_half_width, -plane_half_height, plane_distance), direction.yaw, direction.pitch),
            rotate_local_point((-plane_half_width, -plane_half_height, plane_distance), direction.yaw, direction.pitch),
        ]
        return center_point, corner_points

    def _draw_sphere_wire(
        self,
        radius: float,
        center_x: float,
        center_y: float,
    ) -> None:
        def draw_ring(samples: list[tuple[float, float, float]]) -> None:
            for start_point, end_point in zip(samples, samples[1:]):
                average_depth = (start_point[2] + end_point[2]) / 2.0
                line_color = "#1e293b" if average_depth < 0 else "#64748b"
                line_width = 1 if average_depth < 0 else 2
                dash_pattern = (4, 4) if average_depth < 0 else ()
                self.sphere_canvas.create_line(
                    start_point[0],
                    start_point[1],
                    end_point[0],
                    end_point[1],
                    fill=line_color,
                    width=line_width,
                    dash=dash_pattern,
                )

        def build_latitude_samples(latitude_degrees: float) -> list[tuple[float, float, float]]:
            samples: list[tuple[float, float, float]] = []
            for sample_index in range(0, 73):
                longitude_radians = math.radians(sample_index * 5.0)
                latitude_radians = math.radians(latitude_degrees)
                vector = (
                    math.cos(latitude_radians) * math.sin(longitude_radians),
                    math.sin(latitude_radians),
                    math.cos(latitude_radians) * math.cos(longitude_radians),
                )
                samples.append(self._project_sphere_vector(vector, radius, center_x, center_y))
            return samples

        def build_longitude_samples(longitude_degrees: float) -> list[tuple[float, float, float]]:
            samples: list[tuple[float, float, float]] = []
            for sample_index in range(-18, 19):
                latitude_radians = math.radians(sample_index * 5.0)
                longitude_radians = math.radians(longitude_degrees)
                vector = (
                    math.cos(latitude_radians) * math.sin(longitude_radians),
                    math.sin(latitude_radians),
                    math.cos(latitude_radians) * math.cos(longitude_radians),
                )
                samples.append(self._project_sphere_vector(vector, radius, center_x, center_y))
            return samples

        draw_ring(build_latitude_samples(0.0))
        draw_ring(build_longitude_samples(0.0))
        draw_ring(build_longitude_samples(90.0))

    def _render_direction_sphere(self) -> None:
        self.sphere_canvas.delete("all")
        canvas_width = max(1, self.sphere_canvas.winfo_width())
        canvas_height = max(1, self.sphere_canvas.winfo_height())
        center_x = canvas_width / 2
        center_y = canvas_height / 2
        radius = max(20.0, min(canvas_width, canvas_height) * 0.35)
        output_aspect_ratio = self._preview_aspect_ratio() or (16.0 / 9.0)
        horizontal_fov = safe_positive_float(self.fov_var.get().strip()) or DEFAULT_FOV

        self.sphere_canvas.create_rectangle(0, 0, canvas_width, canvas_height, fill="#0b1220", outline="")
        self.sphere_canvas.create_oval(
            center_x - radius,
            center_y - radius,
            center_x + radius,
            center_y + radius,
            outline="#475569",
            width=2,
        )
        self._draw_sphere_wire(radius, center_x, center_y)

        selected_index = self._selected_direction_index()
        projected_items: list[dict[str, object]] = []
        for index, direction in enumerate(self.directions):
            vector = rotate_local_vector((0.0, 0.0, 1.0), direction.yaw, direction.pitch)
            projected_x, projected_y, depth = self._project_sphere_vector(vector, radius, center_x, center_y)
            color = PREVIEW_COLORS[index % len(PREVIEW_COLORS)]
            plane_center_point, plane_corners = self._build_direction_plane_vertices(
                direction,
                horizontal_fov,
                output_aspect_ratio,
            )
            projected_plane_center = self._project_sphere_vector(plane_center_point, radius, center_x, center_y)
            projected_plane_corners = [
                self._project_sphere_vector(corner_point, radius, center_x, center_y)
                for corner_point in plane_corners
            ]
            projected_items.append(
                {
                    "depth": depth,
                    "index": index,
                    "marker": (projected_x, projected_y),
                    "color": color,
                    "selected": index == selected_index,
                    "plane_center": projected_plane_center,
                    "plane_corners": projected_plane_corners,
                }
            )

        projected_items.sort(key=lambda item: float(item["depth"]))
        index_width = direction_index_width(len(self.directions))
        for item in projected_items:
            depth = float(item["depth"])
            index = int(item["index"])
            projected_x, projected_y = item["marker"]  # type: ignore[index]
            color = str(item["color"])
            is_selected = bool(item["selected"])
            plane_center_x, plane_center_y, plane_center_depth = item["plane_center"]  # type: ignore[index]
            plane_corners = item["plane_corners"]  # type: ignore[assignment]

            if depth < 0:
                fill_color = "#1f2937"
                outline_color = color
            else:
                fill_color = color
                outline_color = "#0b1220"

            line_color = "#334155" if plane_center_depth < 0 else color
            line_width = 1 if not is_selected else 2
            if is_selected:
                self.sphere_canvas.create_line(
                    center_x,
                    center_y,
                    plane_center_x,
                    plane_center_y,
                    fill="white",
                    width=line_width + 2,
                )
            self.sphere_canvas.create_line(
                center_x,
                center_y,
                plane_center_x,
                plane_center_y,
                fill=line_color,
                width=line_width,
                dash=(4, 4) if plane_center_depth < 0 else (),
            )

            flattened_coords: list[float] = []
            for corner_x, corner_y, _corner_depth in plane_corners:
                flattened_coords.extend((corner_x, corner_y))
            if flattened_coords:
                flattened_coords.extend(flattened_coords[:2])
                if is_selected:
                    self.sphere_canvas.create_line(
                        flattened_coords,
                        fill="white",
                        width=line_width + 2,
                    )
                self.sphere_canvas.create_line(
                    flattened_coords,
                    fill=line_color,
                    width=line_width,
                    dash=(4, 4) if plane_center_depth < 0 else (),
                )

            marker_radius = 7 if is_selected else 5
            if is_selected:
                self.sphere_canvas.create_oval(
                    projected_x - (marker_radius + 3),
                    projected_y - (marker_radius + 3),
                    projected_x + (marker_radius + 3),
                    projected_y + (marker_radius + 3),
                    outline="white",
                    width=2,
                )

            self.sphere_canvas.create_oval(
                projected_x - marker_radius,
                projected_y - marker_radius,
                projected_x + marker_radius,
                projected_y + marker_radius,
                fill=fill_color,
                outline=outline_color,
                width=2,
            )
            self.sphere_canvas.create_text(
                projected_x + 10,
                projected_y - 10,
                text=f"{index:0{index_width}d}",
                fill="#e2e8f0" if depth >= 0 else "#94a3b8",
                anchor="sw",
                font=("Consolas", 9, "bold"),
            )

    def _choose_input(self) -> None:
        file_path = filedialog.askopenfilename(
            title="入力動画を選択",
            filetypes=[
                ("Video Files", "*.mp4 *.mov *.insv *.mkv"),
                ("All Files", "*.*"),
            ],
        )
        if not file_path:
            return

        self.input_path_var.set(file_path)
        if not self.output_dir_var.get().strip():
            self.output_dir_var.set(str(Path(file_path).parent / f"{Path(file_path).stem}_frames"))

        self._load_preview_for_current_input()

    def _choose_output(self) -> None:
        directory = filedialog.askdirectory(title="出力フォルダを選択")
        if directory:
            self.output_dir_var.set(directory)

    def _cleanup_preview_file(self) -> None:
        if self.preview_path is None:
            return

        try:
            self.preview_path.unlink(missing_ok=True)
        finally:
            self.preview_path = None

    def _clear_preview(self, info_text: str) -> None:
        self._pause_preview_playback()
        self._close_preview_capture()
        self._set_preview_preparing_state(False)
        self.preview_photo = None
        self.preview_photo_from_snapshot = False
        self.preview_last_frame_bgr = None
        self.preview_metadata = None
        self.preview_box = None
        self._cleanup_preview_file()
        self._reset_preview_controls()
        self.preview_info_var.set(info_text)
        self.preview_selection_var.set("方向を選択すると強調表示されます。")
        self._layout_preview_surface()
        self._render_direction_sphere()

    def _determine_preview_canvas_size(self, metadata: VideoMetadata) -> tuple[int, int]:
        """ウィンドウを画面いっぱいまで広げた場合に確保できるプレビューサイズ。

        戻り値はウィンドウ拡大の希望値であり、実際の描画サイズは
        `_layout_preview_surface` が常に実測値から決める。
        """
        self.root.update_idletasks()

        current_window_width = max(self.root.winfo_width(), self.min_window_width)
        current_window_height = max(self.root.winfo_height(), self.min_window_height)
        media_width, media_height = self._preview_surface_bounds()

        non_preview_width = max(0, current_window_width - media_width)
        non_preview_height = max(0, current_window_height - media_height)

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        max_window_width = max(self.min_window_width, int(screen_width * WINDOW_MAX_SCREEN_WIDTH_RATIO))
        max_window_height = max(self.min_window_height, int(screen_height * WINDOW_MAX_SCREEN_HEIGHT_RATIO))

        max_preview_width = max(
            PREVIEW_SURFACE_MIN_WIDTH,
            max_window_width - non_preview_width,
        )
        max_preview_height = max(
            PREVIEW_SURFACE_MIN_HEIGHT,
            max_window_height - non_preview_height,
        )

        return fit_size_within_bounds(
            metadata.width,
            metadata.height,
            max_preview_width,
            max_preview_height,
            allow_upscale=True,
        )

    def _auto_resize_window_for_preview(self, canvas_width: int, canvas_height: int) -> None:
        """動画が入るようウィンドウを広げる。縮小はせず、画面と最小サイズを必ず守る。"""
        self.root.update_idletasks()

        current_window_width = max(self.root.winfo_width(), self.min_window_width)
        current_window_height = max(self.root.winfo_height(), self.min_window_height)
        media_width, media_height = self._preview_surface_bounds()

        non_preview_width = max(0, current_window_width - media_width)
        non_preview_height = max(0, current_window_height - media_height)

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        max_window_width = max(self.min_window_width, int(screen_width * WINDOW_MAX_SCREEN_WIDTH_RATIO))
        max_window_height = max(self.min_window_height, int(screen_height * WINDOW_MAX_SCREEN_HEIGHT_RATIO))

        target_window_width = min(
            max_window_width,
            max(self.min_window_width, current_window_width, non_preview_width + canvas_width),
        )
        target_window_height = min(
            max_window_height,
            max(self.min_window_height, current_window_height, non_preview_height + canvas_height),
        )

        if target_window_width <= current_window_width and target_window_height <= current_window_height:
            return

        window_x = max(0, self.root.winfo_x())
        window_y = max(0, self.root.winfo_y())
        self.root.geometry(f"{target_window_width}x{target_window_height}+{window_x}+{window_y}")
        self.root.update_idletasks()

    def _layout_preview_surface(self) -> None:
        """プレビュー面を常に表示領域内へ収める。

        動画の有無・ウィンドウサイズ・再生方式にかかわらずここだけがサイズを決めるので、
        どのタイミングで呼んでもレイアウトが壊れない。
        """
        available_width, available_height = self._preview_surface_bounds()

        if self.preview_metadata is None:
            # 動画未選択時は領域全体を使い、案内テキストが必ず見えるようにする。
            target_width, target_height = available_width, available_height
        else:
            target_width, target_height = fit_size_within_bounds(
                self.preview_metadata.width,
                self.preview_metadata.height,
                available_width,
                available_height,
                allow_upscale=True,
            )

        size_changed = (
            target_width != self.preview_canvas_width
            or target_height != self.preview_canvas_height
        )

        self.preview_canvas_width = target_width
        self.preview_canvas_height = target_height
        self.preview_canvas_offset_x = max(0, (available_width - target_width) // 2)
        self.preview_canvas_offset_y = max(0, (available_height - target_height) // 2)
        self._show_preview_video_widget(self._preview_uses_vlc() and self.preview_is_playing)

        if self.preview_metadata is None:
            self.preview_box = None
        else:
            self.preview_box = compute_preview_box(
                self.preview_metadata.width,
                self.preview_metadata.height,
                target_width,
                target_height,
            )

        if size_changed:
            if self.preview_photo_from_snapshot and self._preview_uses_vlc() and not self.preview_is_playing:
                # VLC の静止画は現在位置のものを取り直す (拡大された古い画像を残さない)。
                self._schedule_preview_snapshot(120)
            elif self.preview_last_frame_bgr is not None:
                # 表示済みフレームを新しいサイズで再生成する (再シークしない)。
                self._update_preview_photo(self.preview_last_frame_bgr)
        self._render_preview_overlay()

    def _on_preview_media_configure(self, _event: tk.Event[tk.Misc]) -> None:
        # 動画の有無に関係なく追従させる。metadata が無い間もサイズを合わせないと
        # 案内テキストが表示領域外に描かれてしまう。
        if self.preview_resize_after_id is not None:
            self.root.after_cancel(self.preview_resize_after_id)

        def apply_resize() -> None:
            self.preview_resize_after_id = None
            self._layout_preview_surface()

        self.preview_resize_after_id = self.root.after(30, apply_resize)

    def _load_preview_for_current_input(self, show_errors: bool = True) -> None:
        input_value = self.input_path_var.get().strip()
        if not input_value:
            self._clear_preview("動画を選択するとサムネイルを表示します。")
            return

        input_video = Path(input_value)
        if not input_video.is_file():
            self._clear_preview("入力動画が見つかりません。")
            if show_errors:
                messagebox.showerror("入力エラー", "入力動画が見つかりません。")
            return

        ffmpeg_path = shutil.which("ffmpeg")
        ffprobe_path = shutil.which("ffprobe")
        if ffmpeg_path is None or ffprobe_path is None:
            self._clear_preview("ffmpeg / ffprobe が見つかりません。")
            if show_errors:
                messagebox.showerror("環境エラー", "ffmpeg と ffprobe を PATH に追加してください。")
            return

        self._pause_preview_playback()
        self._close_preview_capture()
        self._cancel_preview_seek()
        self.preview_proxy_request_id += 1
        request_id = self.preview_proxy_request_id
        self.status_var.set("プレビューを読み込んでいます...")
        self.root.update_idletasks()

        try:
            metadata = probe_video_metadata(input_video, ffprobe_path)
            preview_canvas_width, preview_canvas_height = self._determine_preview_canvas_size(metadata)
            initial_frame = self._read_initial_preview_frame(input_video, ffmpeg_path, metadata)
        except Exception as error:
            self._clear_preview("プレビューの読み込みに失敗しました。")
            self._append_log(f"ERROR: {error}")
            self.status_var.set("プレビューの読み込みに失敗しました。")
            if show_errors:
                messagebox.showerror("プレビューエラー", str(error))
            return

        if initial_frame is None:
            self._clear_preview("動画フレームの読み込みに失敗しました。")
            self.status_var.set("プレビューの読み込みに失敗しました。")
            if show_errors:
                messagebox.showerror("プレビューエラー", "動画フレームの読み込みに失敗しました。")
            return

        self.preview_metadata = metadata
        self.preview_current_video_path = input_video
        fast_player_ready = False
        if not self.preview_overlay_sync_var.get():
            fast_player_ready = self._open_preview_player_fast(
                input_video,
                metadata,
                preview_canvas_width,
                preview_canvas_height,
            )
        if not fast_player_ready:
            # VLC を使わないモードでは必ず映像ウィジェットを隠し、Canvas 側に戻す。
            self.preview_mode = "proxy"
            self.preview_total_frames = max(1, int(round((metadata.duration_seconds or 0.0) * PREVIEW_PROXY_FPS_LIMIT)))
            self.preview_duration_seconds = metadata.duration_seconds or 0.0
            self.preview_fps = PREVIEW_PROXY_FPS_LIMIT
            self.preview_current_frame_index = 0
            self._auto_resize_window_for_preview(preview_canvas_width, preview_canvas_height)
            self._layout_preview_surface()
            self.preview_slider_internal_update = True
            self.preview_seek_scale.configure(from_=0, to=max(self.preview_total_frames - 1, 1))
            self.preview_seek_var.set(0.0)
            self.preview_slider_internal_update = False
            self._set_preview_time_for_frame_index(0)

        self._display_preview_frame(initial_frame)
        self.preview_info_var.set(
            f"{input_video.name} | {metadata.width}x{metadata.height} | {format_duration(metadata.duration_seconds)}"
        )
        if fast_player_ready:
            self.status_var.set("プレビューを更新しました。")
            self._set_preview_preparing_state(False)
        else:
            self.status_var.set("軽量プレビューを準備しています...")
            self._set_preview_preparing_state(True)
        self._render_direction_sphere()

        if not fast_player_ready:
            threading.Thread(
                target=self._prepare_preview_proxy_async,
                args=(request_id, input_video, ffmpeg_path, metadata, preview_canvas_width, preview_canvas_height),
                daemon=True,
            ).start()

    def _refresh_direction_table(self, select_index: int | None = None) -> None:
        for item in self.direction_table.get_children():
            self.direction_table.delete(item)

        width = direction_index_width(len(self.directions))
        for index, direction in enumerate(self.directions):
            self.direction_table.insert(
                "",
                "end",
                iid=str(index),
                values=(
                    f"{index:0{width}d}",
                    direction_aware_float(direction.yaw),
                    direction_aware_float(direction.pitch),
                ),
            )

        if select_index is not None and 0 <= select_index < len(self.directions):
            self.direction_table.selection_set(str(select_index))
            self.direction_table.focus(str(select_index))
            self.direction_table.see(str(select_index))
            selected_direction = self.directions[select_index]
            self.yaw_var.set(direction_aware_float(selected_direction.yaw))
            self.pitch_var.set(direction_aware_float(selected_direction.pitch))
            self.preview_selection_var.set(
                f"選択中: #{select_index:0{direction_index_width(len(self.directions))}d} "
                f"(yaw={direction_aware_float(selected_direction.yaw)}, "
                f"pitch={direction_aware_float(selected_direction.pitch)})"
            )
        else:
            self.preview_selection_var.set("方向を選択すると強調表示されます。")

        self._render_preview_overlay()
        self._render_direction_sphere()

    def _selected_direction_index(self) -> int | None:
        selected = self.direction_table.selection()
        if not selected:
            return None
        return int(selected[0])

    def _direction_from_inputs(self) -> Direction:
        return Direction(
            yaw=parse_angle(self.yaw_var.get().strip(), "yaw"),
            pitch=parse_angle(self.pitch_var.get().strip(), "pitch"),
        )

    def _add_direction(self) -> None:
        try:
            direction = self._direction_from_inputs()
        except ValueError as error:
            messagebox.showerror("入力エラー", str(error))
            return

        self.directions.append(direction)
        new_index = len(self.directions) - 1
        self._refresh_direction_table(select_index=new_index)
        self.status_var.set(f"方向を追加しました。現在 {len(self.directions)} 件です。")

    def _update_selected_direction(self) -> None:
        selected_index = self._selected_direction_index()
        if selected_index is None:
            messagebox.showwarning("未選択", "更新する方向を選択してください。")
            return

        try:
            direction = self._direction_from_inputs()
        except ValueError as error:
            messagebox.showerror("入力エラー", str(error))
            return

        self.directions[selected_index] = direction
        self._refresh_direction_table(select_index=selected_index)
        self.status_var.set(f"方向 {selected_index} を更新しました。")

    def _replace_directions(self, directions: list[Direction]) -> None:
        """方向リストを中身ごと差し替える。

        `self.directions` は現在の DirectionSet が持つリストと同一オブジェクトなので、
        再代入すると参照が切れてセット側に反映されず、保存やタブ切り替えで編集が失われる。
        必ずスライス代入で中身だけを入れ替える。
        """
        self.directions[:] = directions

    def _remove_selected_direction(self) -> None:
        selected_index = self._selected_direction_index()
        if selected_index is None:
            messagebox.showwarning("未選択", "削除する方向を選択してください。")
            return

        del self.directions[selected_index]
        if not self.directions:
            self._replace_directions([Direction(yaw=0.0, pitch=0.0)])
            next_index = 0
        else:
            next_index = min(selected_index, len(self.directions) - 1)

        self._refresh_direction_table(select_index=next_index)
        self.status_var.set(f"方向を削除しました。現在 {len(self.directions)} 件です。")

    def _clear_directions(self) -> None:
        self._replace_directions([Direction(yaw=0.0, pitch=0.0)])
        self.yaw_var.set("0")
        self.pitch_var.set("0")
        self._refresh_direction_table(select_index=0)
        self.status_var.set("方向を初期状態に戻しました。")

    def _replace_with_ring(self) -> None:
        try:
            count = parse_positive_int(self.ring_count_var.get().strip(), "方向数")
            pitch = parse_angle(self.ring_pitch_var.get().strip(), "pitch")
        except ValueError as error:
            messagebox.showerror("入力エラー", str(error))
            return

        self._replace_directions(generate_ring_directions(count, pitch))
        self._refresh_direction_table(select_index=0)
        self.status_var.set(f"水平リング {count} 方向で置き換えました。")

    def _on_direction_selected(self, _event: tk.Event[tk.Misc]) -> None:
        selected_index = self._selected_direction_index()
        if selected_index is None:
            self.preview_selection_var.set("方向を選択すると強調表示されます。")
            self._render_preview_overlay()
            return

        direction = self.directions[selected_index]
        self.yaw_var.set(direction_aware_float(direction.yaw))
        self.pitch_var.set(direction_aware_float(direction.pitch))
        self.preview_selection_var.set(
            f"選択中: #{selected_index:0{direction_index_width(len(self.directions))}d} "
            f"(yaw={direction_aware_float(direction.yaw)}, pitch={direction_aware_float(direction.pitch)})"
        )
        self._render_preview_overlay()

    def _preview_aspect_ratio(self) -> float | None:
        output_width = safe_positive_int(self.width_var.get().strip())
        output_height = safe_positive_int(self.height_var.get().strip())
        if output_width is None or output_height is None:
            return None
        return output_width / output_height

    def _render_preview_overlay(self) -> None:
        self.preview_canvas.delete("all")
        self.preview_canvas.create_rectangle(
            0,
            0,
            self.preview_canvas_width,
            self.preview_canvas_height,
            fill="#0f172a",
            outline="",
        )

        if self.preview_photo is not None:
            self.preview_canvas.create_image(0, 0, anchor="nw", image=self.preview_photo)

        if self.preview_metadata is None or self.preview_box is None:
            self.preview_canvas.create_text(
                self.preview_canvas_width / 2,
                self.preview_canvas_height / 2,
                text="入力動画を選択するとサムネイルと方向プレビューを表示します",
                fill="#cbd5e1",
                font=("Yu Gothic UI", 13),
                justify="center",
                # 幅を渡して折り返す。狭いウィンドウでも文字が枠外へ出ない。
                width=max(80, self.preview_canvas_width - 32),
            )
            return

        offset_x, offset_y, display_width, display_height = self.preview_box
        self.preview_canvas.create_rectangle(
            offset_x,
            offset_y,
            offset_x + display_width,
            offset_y + display_height,
            outline="#475569",
            width=1,
        )
        self._draw_preview_grid()

        selected_index = self._selected_direction_index()
        horizontal_fov = safe_positive_float(self.fov_var.get().strip())
        aspect_ratio = self._preview_aspect_ratio()

        for index, direction in enumerate(self.directions):
            color = PREVIEW_COLORS[index % len(PREVIEW_COLORS)]
            is_selected = index == selected_index
            if horizontal_fov is not None and aspect_ratio is not None:
                self._draw_direction_footprint(direction, color, is_selected, horizontal_fov, aspect_ratio)
            self._draw_direction_marker(index, direction, color, is_selected)

    def _draw_preview_grid(self) -> None:
        assert self.preview_metadata is not None
        assert self.preview_box is not None

        meridians = (-180, -90, 0, 90, 180)
        latitudes = (-45, 0, 45)

        for longitude in meridians:
            point_x = ((longitude + 180.0) / 360.0) * self.preview_metadata.width
            canvas_x, _ = map_preview_point(
                (point_x, 0.0),
                self.preview_box,
                self.preview_metadata.width,
                self.preview_metadata.height,
            )
            self.preview_canvas.create_line(
                canvas_x,
                self.preview_box[1],
                canvas_x,
                self.preview_box[1] + self.preview_box[3],
                fill="#334155",
                dash=(4, 4),
            )

        for latitude in latitudes:
            point_y = ((90.0 - latitude) / 180.0) * self.preview_metadata.height
            _, canvas_y = map_preview_point(
                (0.0, point_y),
                self.preview_box,
                self.preview_metadata.width,
                self.preview_metadata.height,
            )
            self.preview_canvas.create_line(
                self.preview_box[0],
                canvas_y,
                self.preview_box[0] + self.preview_box[2],
                canvas_y,
                fill="#334155",
                dash=(4, 4),
            )

    def _draw_direction_marker(
        self,
        index: int,
        direction: Direction,
        color: str,
        is_selected: bool,
    ) -> None:
        assert self.preview_metadata is not None
        assert self.preview_box is not None

        source_point = direction_to_equirect_point(
            direction,
            self.preview_metadata.width,
            self.preview_metadata.height,
        )
        marker_radius = 6 if is_selected else 4
        line_width = 3 if is_selected else 2
        label_width = direction_index_width(len(self.directions))

        for shift in (-self.preview_metadata.width, 0, self.preview_metadata.width):
            shifted_x = source_point[0] + shift
            if shifted_x < -4 or shifted_x > self.preview_metadata.width + 4:
                continue

            canvas_x, canvas_y = map_preview_point(
                (shifted_x, source_point[1]),
                self.preview_box,
                self.preview_metadata.width,
                self.preview_metadata.height,
            )

            if is_selected:
                self.preview_canvas.create_oval(
                    canvas_x - (marker_radius + 3),
                    canvas_y - (marker_radius + 3),
                    canvas_x + (marker_radius + 3),
                    canvas_y + (marker_radius + 3),
                    outline="white",
                    width=2,
                )

            self.preview_canvas.create_oval(
                canvas_x - marker_radius,
                canvas_y - marker_radius,
                canvas_x + marker_radius,
                canvas_y + marker_radius,
                fill=color,
                outline="#0f172a",
                width=line_width,
            )

            label_text = f"{index:0{label_width}d}"
            text_id = self.preview_canvas.create_text(
                canvas_x + 12,
                canvas_y - 12,
                text=label_text,
                fill="white",
                anchor="sw",
                font=("Consolas", 10, "bold"),
            )
            bbox = self.preview_canvas.bbox(text_id)
            if bbox is not None:
                self.preview_canvas.create_rectangle(
                    bbox[0] - 4,
                    bbox[1] - 2,
                    bbox[2] + 4,
                    bbox[3] + 2,
                    fill=color,
                    outline="",
                )
                self.preview_canvas.tag_raise(text_id)

    def _draw_direction_footprint(
        self,
        direction: Direction,
        color: str,
        is_selected: bool,
        horizontal_fov: float,
        aspect_ratio: float,
    ) -> None:
        assert self.preview_metadata is not None
        assert self.preview_box is not None

        segments = build_footprint_segments(
            direction=direction,
            horizontal_fov=horizontal_fov,
            aspect_ratio=aspect_ratio,
            map_width=self.preview_metadata.width,
            map_height=self.preview_metadata.height,
        )
        line_width = 4 if is_selected else 2

        for segment in segments:
            min_x = min(point[0] for point in segment)
            max_x = max(point[0] for point in segment)
            min_shift_index = math.floor((-max_x) / self.preview_metadata.width)
            max_shift_index = math.ceil((self.preview_metadata.width - min_x) / self.preview_metadata.width)

            for shift_index in range(min_shift_index, max_shift_index + 1):
                shift = shift_index * self.preview_metadata.width
                shifted_segment = [(point[0] + shift, point[1]) for point in segment]

                canvas_coords: list[float] = []
                for point in shifted_segment:
                    canvas_x, canvas_y = map_preview_point(
                        point,
                        self.preview_box,
                        self.preview_metadata.width,
                        self.preview_metadata.height,
                    )
                    canvas_coords.extend((canvas_x, canvas_y))

                if len(canvas_coords) >= 4:
                    if is_selected:
                        self.preview_canvas.create_line(
                            canvas_coords,
                            fill="white",
                            width=line_width + 2,
                        )
                    self.preview_canvas.create_line(
                        canvas_coords,
                        fill=color,
                        width=line_width,
                    )

    def _start_processing(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo("処理中", "現在処理中です。完了を待つか停止してください。")
            return

        try:
            options = self._collect_options()
        except ValueError as error:
            messagebox.showerror("入力エラー", str(error))
            return

        output_dir = options["output_dir"]
        video_stem = options["video_stem"]
        run_extract = bool(options["run_extract"])
        run_masks = bool(options["run_masks"])
        assert isinstance(output_dir, Path)
        assert isinstance(video_stem, str | None)

        if run_extract and output_dir.is_dir():
            existing = list(output_dir.glob(build_extracted_image_glob(video_stem)))
            if existing and not messagebox.askyesno(
                "上書き確認",
                f"{len(existing)} 件の既存画像が見つかりました。上書きして続行しますか？",
            ):
                return

        self.stop_requested.clear()
        self._append_log("=== 抽出を開始します ===")
        planned_directions = options["directions"]
        assert isinstance(planned_directions, list)
        parallelism = min(int(options["parallelism"]), len(planned_directions)) if run_extract else 0
        gpu_text = "CUDA ON" if bool(options["use_gpu_decode"]) else "CUDA OFF"
        reverse_text = "逆再生 ON" if bool(options["reverse_video"]) else "逆再生 OFF"
        mode_text = (
            "画像抽出+マスク生成" if run_extract and run_masks
            else "画像抽出のみ" if run_extract
            else "マスク生成のみ"
        )
        mask_text = "SegFormerマスク ON" if run_masks else "SegFormerマスク OFF"
        detail_text = f"細かさ={options['mask_detail_level']}"
        threshold_text = f"しきい値={direction_aware_float(float(options['mask_confidence_threshold']))}"
        mask_parallel_text = f"マスク並列={options['mask_parallelism']}"
        category_text = "対象=" + ",".join(
            MASK_CATEGORY_DISPLAY_NAMES.get(category, category) for category in options["mask_categories"]
        )
        naming_text = "奇数逆順 ON" if bool(options["reverse_direction_index_on_odd"]) else "奇数逆順 OFF"
        self.status_var.set(
            f"{mode_text} を開始しました。"
            f"{gpu_text} / {reverse_text} / {naming_text} / {mask_text} / {mask_parallel_text} / "
            f"{detail_text} / {threshold_text} / {category_text}"
        )
        self.worker_thread = threading.Thread(target=self._run_extraction, args=(options,), daemon=True)
        self.worker_thread.start()

    def _request_stop(self) -> None:
        if not self.worker_thread or not self.worker_thread.is_alive():
            self.status_var.set("停止できる処理はありません。")
            return

        self.stop_requested.set()
        self._terminate_active_processes()
        self.status_var.set("停止要求を送信しました。")

    def _collect_options(self) -> dict[str, object]:
        input_value = self.input_path_var.get().strip()
        output_value = self.output_dir_var.get().strip()
        run_extract = bool(self.run_extract_var.get())
        run_masks = bool(self.generate_masks_var.get())

        if not run_extract and not run_masks:
            raise ValueError("画像抽出かマスク生成のどちらかを選択してください。")
        if not output_value:
            raise ValueError("出力フォルダを指定してください。")

        input_video = Path(input_value) if input_value else None
        output_dir = Path(output_value)

        if run_extract:
            if input_video is None:
                raise ValueError("画像抽出を行う場合は入力動画を指定してください。")
            if not input_video.is_file():
                raise ValueError("入力動画が見つかりません。")
        if run_extract and not self.directions:
            raise ValueError("少なくとも1つの書き出し方向を指定してください。")

        fps = parse_positive_float(self.fps_var.get().strip(), "FPS")
        fov = parse_positive_float(self.fov_var.get().strip(), "画角")
        width = parse_positive_int(self.width_var.get().strip(), "幅")
        height = parse_positive_int(self.height_var.get().strip(), "高さ")
        parallelism = parse_positive_int(self.parallelism_var.get().strip(), "並列数")
        mask_parallelism = parse_positive_int(self.mask_parallelism_var.get().strip(), "マスクバッチ数")
        if mask_parallelism > MASK_BATCH_SIZE_LIMIT:
            raise ValueError(f"マスクバッチ数は1から{MASK_BATCH_SIZE_LIMIT}の範囲で指定してください。")
        reverse_direction_index_on_odd = bool(self.reverse_direction_index_on_odd_var.get())
        mask_confidence_threshold = parse_probability_threshold(
            self.mask_confidence_threshold_var.get().strip(),
            "除外しきい値",
        )
        mask_detail_level = self.mask_detail_level_var.get().strip()
        if mask_detail_level not in MASK_DETAIL_PRESETS:
            raise ValueError("マスクの細かさは標準・高・最高から選択してください。")
        mask_categories = self._selected_mask_categories()

        ffmpeg_path = self.ffmpeg_path or shutil.which("ffmpeg")
        if ffmpeg_path is None:
            raise ValueError("ffmpeg が見つかりません。PATH に追加してから実行してください。")
        self.ffmpeg_path = ffmpeg_path

        if run_masks and not mask_categories:
            raise ValueError("マスク対象を少なくとも1つ選択してください。")

        # 出力フォルダの作成は実処理側で行う。
        # ここで作ると入力エラーや上書き確認のキャンセル時に空フォルダが残る。

        video_stem = input_video.stem if input_video is not None else None
        return {
            "input_video": input_video,
            "output_dir": output_dir,
            "video_stem": video_stem,
            "fps": fps,
            "fov": fov,
            "width": width,
            "height": height,
            "ffmpeg_path": ffmpeg_path,
            "directions": list(self.directions),
            "parallelism": parallelism,
            "mask_parallelism": mask_parallelism,
            "run_extract": run_extract,
            "run_masks": run_masks,
            "use_gpu_decode": bool(self.use_gpu_var.get() and self.cuda_available),
            "reverse_video": bool(self.reverse_var.get()),
            "single_pass": bool(self.single_pass_var.get()),
            "write_xmp": bool(self.write_xmp_var.get()),
            "reverse_direction_index_on_odd": reverse_direction_index_on_odd,
            "generate_masks": bool(self.generate_masks_var.get()),
            "mask_detail_level": mask_detail_level,
            "mask_confidence_threshold": mask_confidence_threshold,
            "mask_categories": mask_categories,
        }

    def _run_extraction(self, options: dict[str, object]) -> None:
        input_video = options["input_video"]
        output_dir = options["output_dir"]
        video_stem = options["video_stem"]
        fps = options["fps"]
        fov = options["fov"]
        width = options["width"]
        height = options["height"]
        ffmpeg_path = options["ffmpeg_path"]
        directions = options["directions"]
        requested_parallelism = options["parallelism"]
        requested_mask_parallelism = options["mask_parallelism"]
        run_extract = options["run_extract"]
        run_masks = options["run_masks"]
        use_gpu_decode = options["use_gpu_decode"]
        reverse_video = options["reverse_video"]
        single_pass = options["single_pass"]
        write_xmp = options["write_xmp"]
        reverse_direction_index_on_odd = options["reverse_direction_index_on_odd"]
        generate_masks = options["generate_masks"]
        mask_detail_level = options["mask_detail_level"]
        mask_confidence_threshold = options["mask_confidence_threshold"]
        mask_categories = options["mask_categories"]

        assert isinstance(input_video, Path | None)
        assert isinstance(output_dir, Path)
        assert isinstance(video_stem, str | None)
        assert isinstance(fps, float)
        assert isinstance(fov, float)
        assert isinstance(width, int)
        assert isinstance(height, int)
        assert isinstance(ffmpeg_path, str)
        assert isinstance(directions, list)
        assert isinstance(requested_parallelism, int)
        assert isinstance(requested_mask_parallelism, int)
        assert isinstance(run_extract, bool)
        assert isinstance(run_masks, bool)
        assert isinstance(use_gpu_decode, bool)
        assert isinstance(reverse_video, bool)
        assert isinstance(single_pass, bool)
        assert isinstance(write_xmp, bool)
        assert isinstance(reverse_direction_index_on_odd, bool)
        assert isinstance(generate_masks, bool)
        assert isinstance(mask_detail_level, str)
        assert isinstance(mask_confidence_threshold, float)
        assert isinstance(mask_categories, list)

        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            total = len(directions)
            parallelism = max(1, min(requested_parallelism, total)) if run_extract else 0
            mask_parallelism = max(1, requested_mask_parallelism)
            gpu_text = "ON" if use_gpu_decode else "OFF"
            reverse_text = "ON" if reverse_video else "OFF"
            naming_text = "ON" if reverse_direction_index_on_odd else "OFF"
            mask_text = "ON" if run_masks else "OFF"
            detail_text = mask_detail_level
            threshold_text = direction_aware_float(mask_confidence_threshold)
            categories_text = ",".join(MASK_CATEGORY_DISPLAY_NAMES.get(category, category) for category in mask_categories)
            mode_text = (
                "画像抽出+マスク生成" if run_extract and run_masks
                else "画像抽出のみ" if run_extract
                else "マスク生成のみ"
            )
            self.log_queue.put(
                (
                    "log",
                    f"実行内容: {mode_text} | 並列数: {parallelism if run_extract else 0} | GPUデコード: {gpu_text} | 逆再生: {reverse_text} | "
                    f"奇数フレーム方向index逆順: {naming_text} | "
                    f"SegFormerマスク: {mask_text} | マスク並列数: {mask_parallelism} | 細かさ: {detail_text} | "
                    f"除外しきい値: {threshold_text} | 対象: {categories_text}",
                )
            )

            if run_extract:
                assert input_video is not None
                assert video_stem is not None

                focal_length_35mm = compute_focal_length_35mm(fov, width, height) if write_xmp else None
                if focal_length_35mm is not None:
                    self.log_queue.put(
                        (
                            "log",
                            f"RealityScan用XMP: 35mm換算焦点距離 {focal_length_35mm:.4f}mm / 歪みなし / 全画像を同一キャリブレーショングループ",
                        )
                    )

                if single_pass:
                    job_status, detail = self._run_single_pass_job(
                        ffmpeg_path=ffmpeg_path,
                        input_video=input_video,
                        output_dir=output_dir,
                        video_stem=video_stem,
                        directions=directions,
                        fps=fps,
                        fov=fov,
                        width=width,
                        height=height,
                        use_gpu_decode=use_gpu_decode,
                        reverse_video=reverse_video,
                        reverse_direction_index_on_odd=reverse_direction_index_on_odd,
                        focal_length_35mm=focal_length_35mm,
                    )
                    if job_status == "error":
                        self.log_queue.put(("error", detail or "抽出に失敗しました。"))
                        return
                    if job_status != "ok" or self.stop_requested.is_set():
                        self.log_queue.put(("status", "停止しました。"))
                        return
                else:
                    if not self._run_direction_pool(
                        ffmpeg_path=ffmpeg_path,
                        input_video=input_video,
                        output_dir=output_dir,
                        video_stem=video_stem,
                        directions=directions,
                        total=total,
                        parallelism=parallelism,
                        fps=fps,
                        fov=fov,
                        width=width,
                        height=height,
                        use_gpu_decode=use_gpu_decode,
                        reverse_video=reverse_video,
                        reverse_direction_index_on_odd=reverse_direction_index_on_odd,
                        focal_length_35mm=focal_length_35mm,
                    ):
                        return

            if run_masks:
                self._generate_segformer_masks(
                    output_dir,
                    video_stem,
                    mask_parallelism,
                    mask_detail_level,
                    mask_confidence_threshold,
                    mask_categories,
                )
                if self.stop_requested.is_set():
                    self.log_queue.put(("status", "停止しました。"))
                    return

            self.log_queue.put(("status", "完了しました。"))
        except Exception as error:
            self.log_queue.put(("error", str(error)))
        finally:
            self._terminate_active_processes()

    def _generate_segformer_masks(
        self,
        output_dir: Path,
        video_stem: str | None,
        parallelism: int,
        detail_level: str,
        confidence_threshold: float,
        selected_categories: list[str],
    ) -> None:
        image_jobs = self._build_mask_image_jobs(output_dir, video_stem)
        if not image_jobs:
            self.log_queue.put(("log", "SegFormerマスク生成: 対象画像が見つかりませんでした。"))
            return

        generator = self._get_segformer_mask_generator()
        self.log_queue.put(
            (
                "log",
                f"SegFormerマスク生成を開始: {len(image_jobs)} 枚 | model={generator.model_id} | "
                f"device={generator.device_label()} | 細かさ={detail_level} | "
                f"除外しきい値={direction_aware_float(confidence_threshold)} | "
                f"対象={','.join(MASK_CATEGORY_DISPLAY_NAMES.get(category, category) for category in selected_categories)}",
            )
        )

        def progress_callback(done_count: int, total_count: int) -> None:
            self.log_queue.put(("log", f"SegFormerマスク生成: {done_count}/{total_count}"))

        generator.generate_masks(
            image_jobs=image_jobs,
            batch_size=parallelism,
            stop_requested=self.stop_requested,
            detail_level=detail_level,
            confidence_threshold=confidence_threshold,
            selected_categories=selected_categories,
            progress_callback=progress_callback,
        )

        if not self.stop_requested.is_set():
            self.log_queue.put(("log", "SegFormerマスク生成が完了しました。"))

    def _finalize_direction_outputs(
        self,
        temp_output_dir: Path,
        output_dir: Path,
        video_stem: str,
        direction_idx: int,
        direction_count: int,
        reverse_direction_index_on_odd: bool,
        focal_length_35mm: float | None = None,
    ) -> None:
        for temp_image_path in sorted(temp_output_dir.glob("*.jpg")):
            if self.stop_requested.is_set():
                return

            try:
                frame_index = int(temp_image_path.stem)
            except ValueError as exc:
                raise RuntimeError(f"一時出力ファイル名の解析に失敗しました: {temp_image_path.name}") from exc

            final_image_path = build_output_image_path(
                output_dir,
                video_stem,
                frame_index,
                direction_idx,
                direction_count,
                reverse_on_odd_frames=reverse_direction_index_on_odd,
            )
            if final_image_path.exists():
                final_image_path.unlink()
            shutil.move(str(temp_image_path), str(final_image_path))

            if focal_length_35mm is not None:
                write_xmp_sidecar(final_image_path, focal_length_35mm)

    def _run_direction_pool(
        self,
        ffmpeg_path: str,
        input_video: Path,
        output_dir: Path,
        video_stem: str,
        directions: list[Direction],
        total: int,
        parallelism: int,
        fps: float,
        fov: float,
        width: int,
        height: int,
        use_gpu_decode: bool,
        reverse_video: bool,
        reverse_direction_index_on_odd: bool,
        focal_length_35mm: float | None,
    ) -> bool:
        """方向ごとに ffmpeg を起動する従来経路。処理を続行してよければ True を返す。"""
        work_queue: queue.Queue[tuple[int, Direction]] = queue.Queue()
        for index, direction in enumerate(directions):
            assert isinstance(direction, Direction)
            work_queue.put((index, direction))

        completed = 0
        completed_lock = threading.Lock()
        errors: list[str] = []
        errors_lock = threading.Lock()

        def worker() -> None:
            nonlocal completed

            while not self.stop_requested.is_set():
                try:
                    index, direction = work_queue.get_nowait()
                except queue.Empty:
                    return

                try:
                    job_status, detail = self._run_direction_job(
                        ffmpeg_path=ffmpeg_path,
                        input_video=input_video,
                        output_dir=output_dir,
                        video_stem=video_stem,
                        direction=direction,
                        index=index,
                        total=total,
                        fps=fps,
                        fov=fov,
                        width=width,
                        height=height,
                        use_gpu_decode=use_gpu_decode,
                        reverse_video=reverse_video,
                        reverse_direction_index_on_odd=reverse_direction_index_on_odd,
                        focal_length_35mm=focal_length_35mm,
                    )

                    if job_status == "ok":
                        with completed_lock:
                            completed += 1
                            self.log_queue.put(("log", f"完了: {completed}/{total}"))
                    elif job_status == "error":
                        with errors_lock:
                            if not errors and detail:
                                errors.append(detail)
                        self.stop_requested.set()
                        self._terminate_active_processes()
                        return
                    else:
                        return
                finally:
                    work_queue.task_done()

        workers = [threading.Thread(target=worker, daemon=True) for _ in range(parallelism)]
        for worker_thread in workers:
            worker_thread.start()
        for worker_thread in workers:
            worker_thread.join()

        if errors:
            self.log_queue.put(("error", errors[0]))
            return False
        if self.stop_requested.is_set():
            self.log_queue.put(("status", "停止しました。"))
            return False
        return True

    def _run_single_pass_job(
        self,
        ffmpeg_path: str,
        input_video: Path,
        output_dir: Path,
        video_stem: str,
        directions: list[Direction],
        fps: float,
        fov: float,
        width: int,
        height: int,
        use_gpu_decode: bool,
        reverse_video: bool,
        reverse_direction_index_on_odd: bool,
        focal_length_35mm: float | None,
    ) -> tuple[str, str | None]:
        """入力を 1 回だけデコードして全方向を同時に書き出す。"""
        total = len(directions)
        label = f"[1パス/{total}方向]"
        self.log_queue.put(("log", f"{label} 単一デコードで全方向を抽出します。"))

        attempts = [use_gpu_decode]
        if use_gpu_decode:
            attempts.append(False)

        with tempfile.TemporaryDirectory(prefix="insta360_extract_") as temp_root_str:
            temp_root = Path(temp_root_str)
            temp_dirs = [temp_root / f"{index:04d}" for index in range(total)]

            for attempt_index, gpu_attempt in enumerate(attempts):
                if self.stop_requested.is_set():
                    return "stopped", None

                for temp_dir in temp_dirs:
                    if temp_dir.exists():
                        shutil.rmtree(temp_dir)
                    temp_dir.mkdir(parents=True)

                if gpu_attempt:
                    self.log_queue.put(("log", f"{label} CUDAデコードで開始"))
                elif use_gpu_decode and attempt_index > 0:
                    self.log_queue.put(("log", f"{label} CUDAデコードに失敗したためCPUデコードで再試行"))

                command = build_single_pass_ffmpeg_command(
                    ffmpeg_path=ffmpeg_path,
                    input_video=input_video,
                    output_patterns=[temp_dir / "%04d.jpg" for temp_dir in temp_dirs],
                    directions=directions,
                    fps=fps,
                    fov=fov,
                    width=width,
                    height=height,
                    use_gpu_decode=gpu_attempt,
                    reverse_video=reverse_video,
                )
                status, detail = self._run_ffmpeg_process(0, label, command)
                if status == "ok":
                    for index, temp_dir in enumerate(temp_dirs):
                        self._finalize_direction_outputs(
                            temp_dir,
                            output_dir,
                            video_stem,
                            index,
                            total,
                            reverse_direction_index_on_odd=reverse_direction_index_on_odd,
                            focal_length_35mm=focal_length_35mm,
                        )
                        self.log_queue.put(("log", f"完了: {index + 1}/{total}"))
                    return "ok", None
                if status == "stopped":
                    return "stopped", None

            return "error", detail

    def _run_direction_job(
        self,
        ffmpeg_path: str,
        input_video: Path,
        output_dir: Path,
        video_stem: str,
        direction: Direction,
        index: int,
        total: int,
        fps: float,
        fov: float,
        width: int,
        height: int,
        use_gpu_decode: bool,
        reverse_video: bool,
        reverse_direction_index_on_odd: bool,
        focal_length_35mm: float | None = None,
    ) -> tuple[str, str | None]:
        even_output_path = build_output_image_path(
            output_dir,
            video_stem,
            0,
            index,
            total,
            reverse_on_odd_frames=False,
        )
        odd_output_path = build_output_image_path(
            output_dir,
            video_stem,
            1,
            index,
            total,
            reverse_on_odd_frames=reverse_direction_index_on_odd,
        )
        label = f"[{index + 1}/{total}]"
        self.log_queue.put(
            (
                "log",
                f"{label} yaw={direction_aware_float(direction.yaw)}, "
                f"pitch={direction_aware_float(direction.pitch)} -> even:{even_output_path.name} odd:{odd_output_path.name}",
            )
        )

        attempts = [use_gpu_decode]
        if use_gpu_decode:
            attempts.append(False)

        with tempfile.TemporaryDirectory(prefix="insta360_extract_") as temp_output_dir_str:
            temp_output_dir = Path(temp_output_dir_str)
            temp_output_pattern = temp_output_dir / "%04d.jpg"

            for attempt_index, gpu_attempt in enumerate(attempts):
                if self.stop_requested.is_set():
                    return "stopped", None

                for temp_image_path in temp_output_dir.glob("*.jpg"):
                    temp_image_path.unlink()

                if gpu_attempt:
                    self.log_queue.put(("log", f"{label} CUDAデコードで開始"))
                elif use_gpu_decode and attempt_index > 0:
                    self.log_queue.put(("log", f"{label} CUDAデコードに失敗したためCPUデコードで再試行"))

                command = build_ffmpeg_command(
                    ffmpeg_path=ffmpeg_path,
                    input_video=input_video,
                    output_dir=output_dir,
                    video_stem=video_stem,
                    direction=direction,
                    direction_idx=index,
                    direction_count=total,
                    fps=fps,
                    fov=fov,
                    width=width,
                    height=height,
                    use_gpu_decode=gpu_attempt,
                    reverse_video=reverse_video,
                    output_pattern_override=temp_output_pattern,
                )
                status, detail = self._run_ffmpeg_process(index, label, command)
                if status == "ok":
                    self._finalize_direction_outputs(
                        temp_output_dir,
                        output_dir,
                        video_stem,
                        index,
                        total,
                        reverse_direction_index_on_odd=reverse_direction_index_on_odd,
                        focal_length_35mm=focal_length_35mm,
                    )
                    return "ok", None
                if status == "stopped":
                    return "stopped", None
                if not gpu_attempt:
                    return "error", detail

        return "error", f"{label} の処理に失敗しました。"

    def _run_ffmpeg_process(
        self,
        process_key: int,
        label: str,
        command: list[str],
    ) -> tuple[str, str | None]:
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=creationflags,
        )
        self._register_process(process_key, process)

        recent_lines: list[str] = []
        try:
            assert process.stdout is not None
            for line in process.stdout:
                if self.stop_requested.is_set() and process.poll() is None:
                    process.terminate()

                cleaned = line.strip()
                if not cleaned:
                    continue

                recent_lines.append(cleaned)
                if len(recent_lines) > 12:
                    recent_lines.pop(0)
                self.log_queue.put(("log", f"{label} {cleaned}"))

            return_code = process.wait()
        finally:
            self._unregister_process(process_key, process)

        if self.stop_requested.is_set():
            return "stopped", None
        if return_code != 0:
            detail = recent_lines[-1] if recent_lines else f"{label} ffmpeg exited with code {return_code}"
            return "error", detail
        return "ok", None

    def _register_process(self, process_key: int, process: subprocess.Popen[str]) -> None:
        with self.process_lock:
            self.current_processes[process_key] = process

    def _unregister_process(self, process_key: int, process: subprocess.Popen[str] | None = None) -> None:
        with self.process_lock:
            current = self.current_processes.get(process_key)
            if current is None:
                return
            if process is None or current is process:
                self.current_processes.pop(process_key, None)

    def _terminate_active_processes(self) -> None:
        with self.process_lock:
            processes = list(self.current_processes.values())

        for process in processes:
            if process.poll() is None:
                process.terminate()

    def _drain_log_queue(self) -> None:
        try:
            while True:
                kind, message = self.log_queue.get_nowait()
                if kind == "log":
                    self._append_log(str(message))
                elif kind == "status":
                    status_text = str(message)
                    self._append_log(status_text)
                    self.status_var.set(status_text)
                    if status_text == "完了しました。":
                        messagebox.showinfo("完了", "画像書き出しが完了しました。")
                    elif status_text == "停止しました。":
                        messagebox.showinfo("停止", "処理を停止しました。")
                elif kind == "error":
                    error_message = str(message)
                    self._append_log(f"ERROR: {error_message}")
                    self.status_var.set("エラーが発生しました。")
                    messagebox.showerror("エラー", error_message)
                elif kind == "preview_proxy_ready":
                    if not isinstance(message, dict):
                        continue
                    request_id = message.get("request_id")
                    proxy_path = message.get("proxy_path")
                    video_path = message.get("video_path")
                    metadata = message.get("metadata")
                    if request_id != self.preview_proxy_request_id:
                        if isinstance(proxy_path, Path):
                            proxy_path.unlink(missing_ok=True)
                        continue
                    if not isinstance(proxy_path, Path) or not isinstance(video_path, Path) or not isinstance(metadata, VideoMetadata):
                        continue
                    try:
                        self._open_preview_capture(proxy_path, video_path, metadata)
                        self._layout_preview_surface()
                        self._render_preview_frame(0, sequential=False)
                        self._set_preview_preparing_state(False)
                        self.status_var.set("プレビューを更新しました。")
                    except Exception as error:
                        self._append_log(f"ERROR: {error}")
                        self._set_preview_preparing_state(False)
                        self.status_var.set("軽量プレビューの準備に失敗しました。")
                elif kind == "preview_proxy_error":
                    if not isinstance(message, dict):
                        continue
                    request_id = message.get("request_id")
                    error_text = message.get("error")
                    if request_id != self.preview_proxy_request_id:
                        continue
                    self._set_preview_preparing_state(False)
                    if isinstance(error_text, str):
                        self._append_log(f"ERROR: {error_text}")
                    self.status_var.set("軽量プレビューの準備に失敗しました。")
        except queue.Empty:
            pass
        finally:
            self.log_drain_after_id = self.root.after(100, self._drain_log_queue)

    def _append_log(self, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _on_close(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            should_close = messagebox.askyesno(
                "確認",
                "処理中です。停止してウィンドウを閉じますか？",
            )
            if not should_close:
                return
            self.stop_requested.set()
            self._terminate_active_processes()

        try:
            self._save_persisted_settings()
        except Exception:
            pass

        self._pause_preview_playback()
        self._close_preview_capture()
        self._release_preview_vlc()
        self._cleanup_preview_file()
        self._cancel_pending_callbacks()
        self.root.destroy()

    def _cancel_pending_callbacks(self) -> None:
        """破棄後に after コールバックが走らないようにまとめて解除する。"""
        self._cancel_preview_playback()
        self._cancel_preview_seek()
        self._cancel_preview_vlc_poll()
        self._cancel_preview_snapshot()
        for attribute in ("log_drain_after_id", "preview_resize_after_id"):
            after_id = getattr(self, attribute, None)
            if after_id is None:
                continue
            try:
                self.root.after_cancel(after_id)
            except Exception:
                pass
            setattr(self, attribute, None)


def main() -> None:
    root = tk.Tk()
    style = ttk.Style(root)
    if "vista" in style.theme_names():
        style.theme_use("vista")
    Insta360ExtractorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
