#!/usr/bin/env python3

import argparse
from itertools import product
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy.optimize import minimize

from filaments import FilamentLibrary, REQUIRED_LAYER_COLORS
from Models import Filament, LayerType


IDEAL_RGB = {
    LayerType.CYAN: np.array([0.0, 255.0, 255.0]),
    LayerType.MAGENTA: np.array([255.0, 0.0, 255.0]),
    LayerType.YELLOW: np.array([255.0, 255.0, 0.0]),
    LayerType.WHITE: np.array([255.0, 255.0, 255.0]),
}

CMY_LAYERS = (LayerType.CYAN, LayerType.MAGENTA, LayerType.YELLOW)
REQUIRED_LAYERS = (*CMY_LAYERS, LayerType.WHITE)


def hex_to_rgb(hex_color: str) -> np.ndarray:
    hex_color = hex_color.strip().lstrip("#")
    if len(hex_color) == 3:
        hex_color = "".join(char * 2 for char in hex_color)
    if len(hex_color) != 6:
        raise ValueError(f"Invalid hex color: {hex_color}")
    return np.array([
        int(hex_color[0:2], 16),
        int(hex_color[2:4], 16),
        int(hex_color[4:6], 16),
    ], dtype=float)


def normalized_rgb_distance(actual: np.ndarray, target: np.ndarray) -> float:
    return float(np.linalg.norm(actual - target) / (np.sqrt(3.0) * 255.0))


def ideal_color_score(filament_set: Dict[LayerType, Filament]) -> Tuple[float, Dict[str, float]]:
    errors = {}
    weighted_errors = []

    for layer_type in REQUIRED_LAYERS:
        actual = hex_to_rgb(filament_set[layer_type].hex_value)
        error = normalized_rgb_distance(actual, IDEAL_RGB[layer_type])
        errors[layer_type.value] = error
        weight = 0.25 if layer_type == LayerType.WHITE else 1.0
        weighted_errors.append(error * weight)

    max_weighted_error = len(CMY_LAYERS) + 0.25
    score = 100.0 * (1.0 - min(sum(weighted_errors) / max_weighted_error, 1.0))
    return score, errors


def achieved_rgb(amounts: np.ndarray, filament_set: Dict[LayerType, Filament]) -> np.ndarray:
    c, m, y, k = np.clip(amounts, 0.0, 1.0)
    cyan_rgb = hex_to_rgb(filament_set[LayerType.CYAN].hex_value) / 255.0
    magenta_rgb = hex_to_rgb(filament_set[LayerType.MAGENTA].hex_value) / 255.0
    yellow_rgb = hex_to_rgb(filament_set[LayerType.YELLOW].hex_value) / 255.0
    white_rgb = hex_to_rgb(filament_set[LayerType.WHITE].hex_value) / 255.0

    achieved = white_rgb.copy() * (1.0 - k)
    achieved *= 1.0 - c * (1.0 - cyan_rgb)
    achieved *= 1.0 - m * (1.0 - magenta_rgb)
    achieved *= 1.0 - y * (1.0 - yellow_rgb)
    return achieved * 255.0


def target_samples(samples_per_axis: int) -> List[np.ndarray]:
    levels = np.linspace(0.0, 255.0, samples_per_axis)
    samples = [
        np.array([r, g, b], dtype=float)
        for r in levels
        for g in levels
        for b in levels
    ]
    samples.extend([
        IDEAL_RGB[LayerType.CYAN],
        IDEAL_RGB[LayerType.MAGENTA],
        IDEAL_RGB[LayerType.YELLOW],
        np.array([255.0, 0.0, 0.0]),
        np.array([0.0, 255.0, 0.0]),
        np.array([0.0, 0.0, 255.0]),
        np.array([128.0, 128.0, 128.0]),
    ])
    return samples


def solve_color_error(target_rgb: np.ndarray, filament_set: Dict[LayerType, Filament]) -> float:
    target_normalized = target_rgb / 255.0
    k0 = 1.0 - np.max(target_normalized)
    denom = max(1.0 - k0, 1e-5)
    c0 = (1.0 - target_normalized[0] - k0) / denom
    m0 = (1.0 - target_normalized[1] - k0) / denom
    y0 = (1.0 - target_normalized[2] - k0) / denom

    initial_guesses = [
        np.clip(np.array([c0, m0, y0, k0]), 0.0, 1.0),
        np.array([0.5, 0.5, 0.5, 0.5]),
        np.array([0.0, 0.0, 0.0, k0]),
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 1.0, 0.0]),
    ]

    def objective(amounts: np.ndarray) -> float:
        rgb = achieved_rgb(amounts, filament_set)
        return float(np.mean(((rgb - target_rgb) / 255.0) ** 2))

    best = min(
        (
            minimize(
                objective,
                x0,
                method="L-BFGS-B",
                bounds=[(0.0, 1.0)] * 4,
                options={"ftol": 1e-10, "maxiter": 300},
            )
            for x0 in initial_guesses
        ),
        key=lambda result: objective(result.x),
    )
    return float(np.sqrt(np.mean((achieved_rgb(best.x, filament_set) - target_rgb) ** 2)))


def gamut_score(
    filament_set: Dict[LayerType, Filament],
    samples_per_axis: int,
    rmse_reference: float,
) -> Tuple[float, float, float, float]:
    errors = np.array([
        solve_color_error(sample, filament_set)
        for sample in target_samples(samples_per_axis)
    ])
    mean_rmse = float(np.mean(errors))
    median_rmse = float(np.median(errors))
    p95_rmse = float(np.percentile(errors, 95))
    score = 100.0 * (1.0 - min(mean_rmse / rmse_reference, 1.0))
    return score, mean_rmse, median_rmse, p95_rmse


def td_score(
    filament_set: Dict[LayerType, Filament],
    cym_target_thickness: float,
    white_target_thickness: float,
    height_step: float,
    target_cmy_levels: float,
    target_white_levels: float,
) -> Tuple[float, Dict[str, float]]:
    cmy_heights = {
        layer_type.value: filament_set[layer_type].transmission_distance * cym_target_thickness
        for layer_type in CMY_LAYERS
    }
    white_height = filament_set[LayerType.WHITE].transmission_distance * white_target_thickness

    if height_step > 0:
        cmy_levels = {name: height / height_step for name, height in cmy_heights.items()}
        white_levels = white_height / height_step
        cmy_level_score = float(np.mean([
            min(levels / target_cmy_levels, 1.0)
            for levels in cmy_levels.values()
        ]))
        white_level_score = min(white_levels / target_white_levels, 1.0)
    else:
        cmy_levels = {name: float("inf") for name in cmy_heights}
        white_levels = float("inf")
        cmy_level_score = 1.0
        white_level_score = 1.0

    min_cmy_height = min(cmy_heights.values())
    max_cmy_height = max(cmy_heights.values())
    balance_score = min_cmy_height / max_cmy_height if max_cmy_height > 0 else 0.0

    score = 100.0 * (
        0.65 * cmy_level_score
        + 0.20 * white_level_score
        + 0.15 * balance_score
    )
    details = {
        "cyan_height_mm": cmy_heights["cyan"],
        "magenta_height_mm": cmy_heights["magenta"],
        "yellow_height_mm": cmy_heights["yellow"],
        "white_height_mm": white_height,
        "cyan_levels": cmy_levels["cyan"],
        "magenta_levels": cmy_levels["magenta"],
        "yellow_levels": cmy_levels["yellow"],
        "white_levels": white_levels,
        "cmy_balance": balance_score,
    }
    return score, details


def labels_to_score(library: FilamentLibrary, requested_labels: Iterable[str] | None) -> List[str]:
    if requested_labels:
        return list(requested_labels)
    return library.available_labels()


def filament_details(
    filament_set: Dict[LayerType, Filament],
    source_ids: Dict[LayerType, str],
) -> Dict:
    return {
        layer_type.value: {
            "id": source_ids.get(layer_type),
            "manufacturer": filament_set[layer_type].manufacturer,
            "color_name": filament_set[layer_type].color_name,
            "hex_value": filament_set[layer_type].hex_value,
            "transmission_distance": filament_set[layer_type].transmission_distance,
            "label": filament_set[layer_type].label,
        }
        for layer_type in REQUIRED_LAYERS
    }


def score_set(
    name: str,
    filament_set: Dict[LayerType, Filament],
    source_ids: Dict[LayerType, str],
    args: argparse.Namespace,
    kind: str,
) -> Dict:
    ideal_score_value, ideal_errors = ideal_color_score(filament_set)
    gamut_score_value, mean_rmse, median_rmse, p95_rmse = gamut_score(
        filament_set,
        args.samples_per_axis,
        args.rmse_reference,
    )
    td_score_value, td_details = td_score(
        filament_set,
        args.cym_target_thickness,
        args.white_target_thickness,
        args.height_step,
        args.target_cmy_levels,
        args.target_white_levels,
    )
    overall_score = (
        args.ideal_weight * ideal_score_value
        + args.gamut_weight * gamut_score_value
        + args.td_weight * td_score_value
    ) / (args.ideal_weight + args.gamut_weight + args.td_weight)

    return {
        "name": name,
        "kind": kind,
        "overall_score": overall_score,
        "ideal_score": ideal_score_value,
        "gamut_score": gamut_score_value,
        "td_score": td_score_value,
        "mean_rmse": mean_rmse,
        "median_rmse": median_rmse,
        "p95_rmse": p95_rmse,
        "ideal_errors": ideal_errors,
        "td": td_details,
        "filaments": filament_details(filament_set, source_ids),
    }


def source_ids_for_labeled_set(
    filament_set: Dict[LayerType, Filament],
    library: FilamentLibrary,
) -> Dict[LayerType, str]:
    source_ids = {}
    for layer_type, filament in filament_set.items():
        for filament_id, candidate in library.filaments.items():
            if candidate == filament:
                source_ids[layer_type] = filament_id
                break
    return source_ids


def score_label(label: str, library: FilamentLibrary, args: argparse.Namespace) -> Dict:
    filament_set = library.get_filament_set(label)
    source_ids = source_ids_for_labeled_set(filament_set, library)
    return score_set(label, filament_set, source_ids, args, "label")


def layer_candidates(
    library: FilamentLibrary,
    labels: Iterable[str] | None,
) -> Dict[LayerType, List[Tuple[str, Filament]]]:
    normalized_labels = {label.strip().lower() for label in labels} if labels else None
    candidates = {layer_type: [] for layer_type in REQUIRED_LAYERS}

    for filament_id, filament in library.filaments.items():
        color_name = filament.color_name.strip().lower()
        if color_name not in REQUIRED_LAYER_COLORS:
            continue

        if normalized_labels is not None:
            filament_labels = {label.strip().lower() for label in filament.label}
            if not filament_labels & normalized_labels:
                continue

        layer_type = REQUIRED_LAYER_COLORS[color_name]
        if layer_type in candidates:
            candidates[layer_type].append((filament_id, filament))

    missing = [layer_type.value for layer_type, items in candidates.items() if not items]
    if missing:
        raise ValueError(f"No candidate filament(s) for layer(s): {', '.join(missing)}")

    return candidates


def compact_id(filament_id: str, layer_type: LayerType) -> str:
    suffixes = (
        f"_{layer_type.value}_pla",
        f"_{layer_type.value}",
        "_pla",
    )
    for suffix in suffixes:
        if filament_id.endswith(suffix):
            return filament_id[:-len(suffix)]
    return filament_id


def mix_name(source_ids: Dict[LayerType, str]) -> str:
    return (
        f"c={compact_id(source_ids[LayerType.CYAN], LayerType.CYAN)},"
        f"m={compact_id(source_ids[LayerType.MAGENTA], LayerType.MAGENTA)},"
        f"y={compact_id(source_ids[LayerType.YELLOW], LayerType.YELLOW)},"
        f"w={compact_id(source_ids[LayerType.WHITE], LayerType.WHITE)}"
    )


def mix_sets(
    library: FilamentLibrary,
    labels: Iterable[str] | None,
) -> List[Tuple[str, Dict[LayerType, Filament], Dict[LayerType, str]]]:
    candidates = layer_candidates(library, labels)
    combinations = []

    for cyan, magenta, yellow, white in product(
        candidates[LayerType.CYAN],
        candidates[LayerType.MAGENTA],
        candidates[LayerType.YELLOW],
        candidates[LayerType.WHITE],
    ):
        source_ids = {
            LayerType.CYAN: cyan[0],
            LayerType.MAGENTA: magenta[0],
            LayerType.YELLOW: yellow[0],
            LayerType.WHITE: white[0],
        }
        filament_set = {
            LayerType.CYAN: cyan[1],
            LayerType.MAGENTA: magenta[1],
            LayerType.YELLOW: yellow[1],
            LayerType.WHITE: white[1],
        }
        combinations.append((mix_name(source_ids), filament_set, source_ids))

    return combinations


def print_table(results: List[Dict]) -> None:
    name_width = 48
    header = (
        "set",
        "overall",
        "ideal",
        "gamut",
        "td",
        "mean_rmse",
        "p95_rmse",
        "c/m/y max mm",
        "c/m/y levels",
        "white mm",
    )
    print(
        f"{header[0]:<{name_width}} {header[1]:>8} {header[2]:>8} {header[3]:>8} "
        f"{header[4]:>8} {header[5]:>10} {header[6]:>10} "
        f"{header[7]:>22} {header[8]:>22} {header[9]:>10}"
    )
    print("-" * (name_width + 126))

    for result in results:
        td = result["td"]
        heights = f"{td['cyan_height_mm']:.3f}/{td['magenta_height_mm']:.3f}/{td['yellow_height_mm']:.3f}"
        levels = f"{td['cyan_levels']:.1f}/{td['magenta_levels']:.1f}/{td['yellow_levels']:.1f}"
        display_name = result["name"]
        if len(display_name) > name_width:
            display_name = display_name[:name_width - 3] + "..."
        print(
            f"{display_name:<{name_width}} "
            f"{result['overall_score']:>8.1f} "
            f"{result['ideal_score']:>8.1f} "
            f"{result['gamut_score']:>8.1f} "
            f"{result['td_score']:>8.1f} "
            f"{result['mean_rmse']:>10.1f} "
            f"{result['p95_rmse']:>10.1f} "
            f"{heights:>22} "
            f"{levels:>22} "
            f"{td['white_height_mm']:>10.3f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank complete CMYK filament sets for color mixing")
    parser.add_argument("--filaments", default="filaments.yaml", help="Path to filament YAML file")
    parser.add_argument("--label", action="append", dest="labels",
                        help="Only score this label, or restrict mix candidates to this label. Can be passed multiple times")
    parser.add_argument("--mix-and-match", action="store_true",
                        help="Search all cyan/magenta/yellow/white combinations across candidate filaments")
    parser.add_argument("--top", type=int, default=10,
                        help="Number of top rows to print")
    parser.add_argument("--samples-per-axis", type=int, default=5,
                        help="RGB grid density for gamut scoring")
    parser.add_argument("--height-step", type=float, default=0.1,
                        help="Layer-height quantization step in mm")
    parser.add_argument("--cym-target-thickness", type=float, default=0.07,
                        help="CMY thickness multiplier used by the pipeline")
    parser.add_argument("--white-target-thickness", type=float, default=0.16,
                        help="White/intensity thickness multiplier used by the pipeline")
    parser.add_argument("--target-cmy-levels", type=float, default=5.0,
                        help="Desired number of printable height steps per CMY channel")
    parser.add_argument("--target-white-levels", type=float, default=8.0,
                        help="Desired number of printable height steps for white/intensity")
    parser.add_argument("--rmse-reference", type=float, default=128.0,
                        help="Mean RGB RMSE that maps the gamut score to zero")
    parser.add_argument("--ideal-weight", type=float, default=0.35,
                        help="Weight for closeness to ideal CMY/white colors")
    parser.add_argument("--gamut-weight", type=float, default=0.45,
                        help="Weight for simulated RGB gamut fit")
    parser.add_argument("--td-weight", type=float, default=0.20,
                        help="Weight for TD/printable-height headroom")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples_per_axis < 2:
        raise ValueError("--samples-per-axis must be at least 2")
    if args.ideal_weight < 0 or args.gamut_weight < 0 or args.td_weight < 0:
        raise ValueError("Weights must be non-negative")
    if args.ideal_weight + args.gamut_weight + args.td_weight <= 0:
        raise ValueError("At least one score weight must be positive")

    library = FilamentLibrary.from_yaml(Path(args.filaments))
    results = []
    skipped = []

    if args.mix_and_match:
        for name, filament_set, source_ids in mix_sets(library, args.labels):
            results.append(score_set(name, filament_set, source_ids, args, "mix"))
    else:
        for label in labels_to_score(library, args.labels):
            try:
                results.append(score_label(label, library, args))
            except ValueError as error:
                skipped.append({"label": label, "reason": str(error)})

    results.sort(key=lambda result: result["overall_score"], reverse=True)
    printed_results = results[:args.top] if args.top > 0 else results

    if args.json:
        print(json.dumps({"results": printed_results, "skipped": skipped}, indent=2))
        return

    print_table(printed_results)
    if skipped:
        print("\nSkipped labels:")
        for item in skipped:
            print(f"- {item['label']}: {item['reason']}")
    if args.mix_and_match and printed_results:
        best = printed_results[0]
        print("\nBest mix:")
        for layer_type in REQUIRED_LAYERS:
            details = best["filaments"][layer_type.value]
            print(
                f"- {layer_type.value}: {details['id']} "
                f"({details['hex_value']}, TD={details['transmission_distance']})"
            )

    print("\nNotes:")
    print("- ideal: closeness to ideal cyan/magenta/yellow/white RGB values.")
    print("- gamut: simulated RGB mixing fit across a coarse RGB target grid; higher is better.")
    print("- td: heuristic printable-height headroom from transmission distance and layer quantization.")
    print("- This is a model ranking, not a substitute for camera calibration.")


if __name__ == "__main__":
    main()
