from pydantic import BaseModel
from typing import Dict, Optional
import yaml
from pathlib import Path
from Models import Filament, LayerType


REQUIRED_LAYER_COLORS = {
    "cyan": LayerType.CYAN,
    "yellow": LayerType.YELLOW,
    "magenta": LayerType.MAGENTA,
    "white": LayerType.WHITE,
    "k": LayerType.WHITE,
    "key": LayerType.WHITE,
    "black": LayerType.WHITE,
}


class FilamentLibrary(BaseModel):
    filaments: dict[str, Filament]

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "FilamentLibrary":
        with open(yaml_path, 'r') as file:
            data = yaml.safe_load(file)
        return cls(filaments=data)

    def get_filament(self, filament_id: str) -> Optional[Filament]:
        return self.filaments.get(filament_id)

    def available_labels(self) -> list[str]:
        return sorted({
            label
            for filament in self.filaments.values()
            for label in filament.label
        })

    def get_filament_set(self, label: str) -> Dict[LayerType, Filament]:
        normalized_label = label.strip().lower()
        matches = {
            filament_id: filament
            for filament_id, filament in self.filaments.items()
            if normalized_label in {filament_label.strip().lower() for filament_label in filament.label}
        }

        if not matches:
            available = ", ".join(self.available_labels()) or "none"
            raise ValueError(f"No filaments found for label '{label}'. Available labels: {available}")

        filament_set: Dict[LayerType, Filament] = {}
        source_ids: Dict[LayerType, str] = {}

        for filament_id, filament in matches.items():
            color_name = filament.color_name.strip().lower()
            if color_name not in REQUIRED_LAYER_COLORS:
                continue

            layer_type = REQUIRED_LAYER_COLORS[color_name]
            if layer_type in filament_set:
                previous_id = source_ids[layer_type]
                raise ValueError(
                    f"Label '{label}' has multiple {layer_type.value} filaments: "
                    f"{previous_id}, {filament_id}"
                )

            filament_set[layer_type] = filament
            source_ids[layer_type] = filament_id

        required_layers = [LayerType.CYAN, LayerType.YELLOW, LayerType.MAGENTA, LayerType.WHITE]
        missing = [layer_type.value for layer_type in required_layers if layer_type not in filament_set]
        if missing:
            raise ValueError(
                f"Label '{label}' is missing required CMYK filament(s): {', '.join(missing)}"
            )

        return filament_set
