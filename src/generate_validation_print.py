#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np

from Models import LayerType, LuminanceConfig, StlConfig, StlCollection
from filaments import FilamentLibrary
from to_stl import create_base_plate, create_color_layer


PRIMES = (2, 3, 5, 7)


def van_der_corput(index: int, base: int) -> float:
    result = 0.0
    denominator = 1.0

    while index > 0:
        index, remainder = divmod(index, base)
        denominator *= base
        result += remainder / denominator

    return result


def sample_amount(row: int, col: int, patches_x: int) -> Tuple[str, Tuple[float, float, float, float]]:
    ramp_value = col / (patches_x - 1) if patches_x > 1 else 0.0

    if row == 0:
        return "k_ramp", (0.0, 0.0, 0.0, ramp_value)
    if row == 1:
        return "c_ramp", (ramp_value, 0.0, 0.0, 0.0)
    if row == 2:
        return "m_ramp", (0.0, ramp_value, 0.0, 0.0)
    if row == 3:
        return "y_ramp", (0.0, 0.0, ramp_value, 0.0)

    sample_index = ((row - 4) * patches_x) + col + 1
    return "halton_cmyk", tuple(van_der_corput(sample_index, base) for base in PRIMES)


def layer_thicknesses(
    amounts: Tuple[float, float, float, float],
    config: StlConfig,
) -> Dict[str, float]:
    c, m, y, k = amounts
    return {
        "cyan": c * config.luminance_config.cym_target_thickness * config.filament_library[LayerType.CYAN].transmission_distance,
        "magenta": m * config.luminance_config.cym_target_thickness * config.filament_library[LayerType.MAGENTA].transmission_distance,
        "yellow": y * config.luminance_config.cym_target_thickness * config.filament_library[LayerType.YELLOW].transmission_distance,
        "white_requested": k * config.luminance_config.white_target_thickness * config.filament_library[LayerType.WHITE].transmission_distance,
        "white_printed": max(
            k * config.luminance_config.white_target_thickness * config.filament_library[LayerType.WHITE].transmission_distance,
            config.intensity_min_height,
        ),
    }


def fill_region(
    maps: Dict[LayerType, np.ndarray],
    x0: int,
    y0: int,
    width: int,
    height: int,
    thicknesses: Dict[str, float],
) -> None:
    region = np.s_[y0:y0 + height, x0:x0 + width]
    maps[LayerType.CYAN][region] = thicknesses["cyan"]
    maps[LayerType.MAGENTA][region] = thicknesses["magenta"]
    maps[LayerType.YELLOW][region] = thicknesses["yellow"]
    maps[LayerType.WHITE][region] = thicknesses["white_requested"]


def bounds_mm(x0: int, y0: int, width: int, height: int, resolution: float) -> Dict[str, float]:
    return {
        "x": x0 * resolution,
        "y": y0 * resolution,
        "width": width * resolution,
        "height": height * resolution,
    }


def build_preview(
    amounts_map: np.ndarray,
    output_path: Path,
) -> None:
    # This is a visual index only, not a color-accurate simulation.
    c = amounts_map[:, :, 0]
    m = amounts_map[:, :, 1]
    y = amounts_map[:, :, 2]
    k = amounts_map[:, :, 3]
    rgb = np.ones((*amounts_map.shape[:2], 3), dtype=float)
    rgb[:, :, 0] *= (1.0 - c) * (1.0 - k)
    rgb[:, :, 1] *= (1.0 - m) * (1.0 - k)
    rgb[:, :, 2] *= (1.0 - y) * (1.0 - k)
    preview = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(output_path), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))


def generate_validation_print(args: argparse.Namespace) -> None:
    output_dir = Path(args.output)
    os.makedirs(output_dir, exist_ok=True)

    width_px = int(round(args.total_width / args.resolution))
    border_px = int(round(args.border / args.resolution))
    marker_px = int(round(args.marker_size / args.resolution))

    if width_px <= 0:
        raise ValueError("Total width must be positive")
    if border_px <= 0:
        raise ValueError("Border must be at least one pixel at the selected resolution")
    if marker_px <= 0:
        raise ValueError("Marker size must be at least one pixel at the selected resolution")
    if width_px <= border_px * 2:
        raise ValueError("Total width is too small for the requested border")

    inner_px = width_px - (2 * border_px)
    patch_px = min(inner_px // args.patches_x, inner_px // args.patches_y)

    if patch_px <= 0:
        raise ValueError("Total width/resolution cannot fit the requested patch grid")

    grid_width_px = patch_px * args.patches_x
    grid_height_px = patch_px * args.patches_y
    grid_x0 = border_px + ((inner_px - grid_width_px) // 2)
    grid_y0 = border_px + ((inner_px - grid_height_px) // 2)

    yaml_path = Path(args.filaments)
    library = FilamentLibrary.from_yaml(yaml_path)
    filament_library = library.get_filament_set(args.filament_label) if args.filament_label else {
        LayerType.CYAN: library.get_filament(args.cyan_filament),
        LayerType.YELLOW: library.get_filament(args.yellow_filament),
        LayerType.MAGENTA: library.get_filament(args.magenta_filament),
        LayerType.WHITE: library.get_filament(args.white_filament),
    }
    config = StlConfig(
        pixel_size=args.resolution,
        base_height=args.base_height,
        intensity_min_height=args.white_min_height,
        height_step_mm=args.height_step,
        face_up=args.face_up,
        include_clear_filler=args.with_clear,
        luminance_config=LuminanceConfig(
            cym_target_thickness=args.cym_target_thickness,
            white_target_thickness=args.white_target_thickness,
        ),
        filament_library=filament_library,
    )

    for layer_type, filament in config.filament_library.items():
        if filament is None:
            raise ValueError(f"Missing filament for layer {layer_type.value}")

    maps = {
        LayerType.CYAN: np.zeros((width_px, width_px), dtype=float),
        LayerType.YELLOW: np.zeros((width_px, width_px), dtype=float),
        LayerType.MAGENTA: np.zeros((width_px, width_px), dtype=float),
        LayerType.WHITE: np.zeros((width_px, width_px), dtype=float),
    }
    amounts_map = np.zeros((width_px, width_px, 4), dtype=float)

    patches = []
    for row in range(args.patches_y):
        for col in range(args.patches_x):
            patch_type, amounts = sample_amount(row, col, args.patches_x)
            thicknesses = layer_thicknesses(amounts, config)
            x0 = grid_x0 + (col * patch_px)
            y0 = grid_y0 + (row * patch_px)
            fill_region(maps, x0, y0, patch_px, patch_px, thicknesses)
            amounts_map[y0:y0 + patch_px, x0:x0 + patch_px] = amounts
            patches.append({
                "index": (row * args.patches_x) + col,
                "row": row,
                "col": col,
                "type": patch_type,
                "c": amounts[0],
                "m": amounts[1],
                "y": amounts[2],
                "k": amounts[3],
                "bounds_px": {"x": x0, "y": y0, "width": patch_px, "height": patch_px},
                "bounds_mm": bounds_mm(x0, y0, patch_px, patch_px, args.resolution),
                "layer_thickness_mm": thicknesses,
            })

    marker_amounts = (1.0, 1.0, 1.0, 1.0)
    marker_thicknesses = layer_thicknesses(marker_amounts, config)
    marker_px = min(marker_px, border_px)
    large_marker_px = min(border_px, marker_px + max(1, marker_px // 2))
    marker_inset_px = max(0, (border_px - marker_px) // 2)
    large_marker_inset_px = max(0, (border_px - large_marker_px) // 2)
    marker_specs = [
        ("top_left_orientation", large_marker_inset_px, large_marker_inset_px, large_marker_px),
        ("top_right", width_px - marker_inset_px - marker_px, marker_inset_px, marker_px),
        ("bottom_left", marker_inset_px, width_px - marker_inset_px - marker_px, marker_px),
        ("bottom_right", width_px - marker_inset_px - marker_px, width_px - marker_inset_px - marker_px, marker_px),
    ]

    markers = []
    for name, x0, y0, size_px in marker_specs:
        fill_region(maps, x0, y0, size_px, size_px, marker_thicknesses)
        amounts_map[y0:y0 + size_px, x0:x0 + size_px] = marker_amounts
        markers.append({
            "name": name,
            "c": marker_amounts[0],
            "m": marker_amounts[1],
            "y": marker_amounts[2],
            "k": marker_amounts[3],
            "bounds_px": {"x": x0, "y": y0, "width": size_px, "height": size_px},
            "bounds_mm": bounds_mm(x0, y0, size_px, size_px, args.resolution),
            "layer_thickness_mm": marker_thicknesses,
        })

    print("creating stl: white_base_mesh.stl")
    base_mesh = create_base_plate(width_px, width_px, config)
    base_heights = np.full((width_px, width_px), config.base_height, dtype=float)
    previous_heights = base_heights
    meshes = {"white_base_mesh": base_mesh}

    layers = [
        ("cyan_mesh", maps[LayerType.CYAN], LayerType.CYAN),
        ("yellow_mesh", maps[LayerType.YELLOW], LayerType.YELLOW),
        ("magenta_mesh", maps[LayerType.MAGENTA], LayerType.MAGENTA),
    ]
    if config.include_clear_filler:
        layers.append(("clear_mesh", maps[LayerType.WHITE], LayerType.CLEAR))
    layers.append(("white_intensity_mesh", maps[LayerType.WHITE], LayerType.WHITE))

    for name, height_map, layer_type in layers:
        print(f"creating stl: {name}.stl")
        mesh, previous_heights = create_color_layer(
            height_map=height_map,
            previous_heights=previous_heights,
            config=config,
            layer_type=layer_type,
            flat_top=layer_type == LayerType.CLEAR,
        )
        meshes[name] = mesh

    StlCollection(meshes=meshes).save_to_folder(output_dir)

    metadata = {
        "description": "CMYK lithophane validation chart",
        "requested_total_width_mm": args.total_width,
        "actual_total_width_mm": width_px * args.resolution,
        "resolution_mm": args.resolution,
        "pixel_count": width_px,
        "border_mm": border_px * args.resolution,
        "patches_x": args.patches_x,
        "patches_y": args.patches_y,
        "patch_size_px": patch_px,
        "patch_size_mm": patch_px * args.resolution,
        "grid_bounds_px": {"x": grid_x0, "y": grid_y0, "width": grid_width_px, "height": grid_height_px},
        "grid_bounds_mm": bounds_mm(grid_x0, grid_y0, grid_width_px, grid_height_px, args.resolution),
        "include_clear_filler": config.include_clear_filler,
        "layer_order": list(meshes.keys()),
        "config": config.model_dump(mode="json"),
        "markers": markers,
        "patches": patches,
    }

    metadata_path = output_dir / "validation_chart_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    build_preview(amounts_map, output_dir / "validation_chart_preview.png")

    print(f"saved {len(meshes)} STL files to {output_dir}")
    print(f"saved metadata to {metadata_path}")
    print(f"actual dimensions: {width_px * args.resolution:.2f}mm x {width_px * args.resolution:.2f}mm")
    print(f"patch grid: {args.patches_x}x{args.patches_y}, {patch_px * args.resolution:.2f}mm patches")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a CMYK lithophane validation chart")
    parser.add_argument("--total-width", "--width", "-w", dest="total_width", type=float, default=60.0,
                        help="Total outside width of the square validation print in mm")
    parser.add_argument("--resolution", "-r", type=float, default=0.4,
                        help="Resolution in mm per generated pixel/block")
    parser.add_argument("--output", "-o", default="stl-output/validation-print",
                        help="Output directory for STL files and metadata")
    parser.add_argument("--border", type=float, default=4.0,
                        help="Border width in mm")
    parser.add_argument("--marker-size", type=float, default=2.0,
                        help="Registration marker size in mm")
    parser.add_argument("--patches-x", type=int, default=10,
                        help="Number of validation patches across")
    parser.add_argument("--patches-y", type=int, default=10,
                        help="Number of validation patches down")
    parser.add_argument("--with-clear", action="store_true", default=False,
                        help="Include a clear filler layer before the white intensity layer")
    parser.add_argument("--face-up", action="store_true", default=False,
                        help="Mirror STLs for face-up viewing")
    parser.add_argument("--filaments", default="filaments.yaml",
                        help="Path to filament YAML file")
    parser.add_argument("--filament-label", default=None,
                        help="Select a complete CMYK filament set from the filament YAML by label")
    parser.add_argument("--cyan-filament", default="bambu_cyan_pla",
                        help="Filament ID for cyan layer")
    parser.add_argument("--yellow-filament", default="bambu_yellow_pla",
                        help="Filament ID for yellow layer")
    parser.add_argument("--magenta-filament", default="bambu_magenta_pla",
                        help="Filament ID for magenta layer")
    parser.add_argument("--white-filament", default="bambu_white_pla",
                        help="Filament ID for white/intensity layer")
    parser.add_argument("--base-height", type=float, default=0.2,
                        help="Base plate height in mm")
    parser.add_argument("--white-min-height", type=float, default=0.2,
                        help="Minimum white/intensity layer height in mm")
    parser.add_argument("--height-step", type=float, default=0.1,
                        help="Height quantization step in mm")
    parser.add_argument("--cym-target-thickness", type=float, default=0.07,
                        help="Target thickness multiplier for CMY layers")
    parser.add_argument("--white-target-thickness", type=float, default=0.16,
                        help="Target thickness multiplier for white/intensity layer")
    return parser.parse_args()


if __name__ == "__main__":
    generate_validation_print(parse_args())
