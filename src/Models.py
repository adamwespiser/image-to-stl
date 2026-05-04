from dataclasses import dataclass
import numpy as np
from stl.mesh import Mesh
from typing import Dict, Tuple
from pydantic import BaseModel, Field, field_validator
from enum import Enum
import os

class LuminanceConfig(BaseModel):
    cym_target_thickness: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Target thickness of the cyan layer in mm"
    )
    white_target_thickness: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Target thickness of the cyan layer in mm"
    )

class LayerType(Enum):
    CYAN = "cyan"
    YELLOW = "yellow"
    MAGENTA = "magenta"
    WHITE = "white"  # For intensity/black
    CLEAR = "clear"  # For intensity/black
    BASE = "base"

class Filament(BaseModel):
    manufacturer: str
    filament_type: str
    color_name: str
    hex_value: str
    transmission_distance: float
    label: list[str] = Field(default_factory=list)

    @field_validator("label", mode="before")
    @classmethod
    def normalize_label(cls, value):
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return value

@dataclass
class FilamentProperties:
    """Properties of a specific filament color"""
    rgb: Tuple[float, float, float]  # Measured RGB values (0-1)
    max_distance: float              # Maximum transmission distance
    min_thickness: float            # Minimum printable thickness

class IntensityChannels(BaseModel):
    c_channel: np.ndarray
    y_channel: np.ndarray
    m_channel: np.ndarray
    intensity_map: np.ndarray

    model_config = {
        'arbitrary_types_allowed': True
    }

class ColorCorrection(Enum):
    LINEAR = "linear"
    LUMINANCE = "luminance"

class StlConfig(BaseModel):
    """Configuration for STL generation"""
    base_height: float = Field(default=0.2, description="Height of the base plate in mm")
    pixel_size: float = Field(default=1.0, description="Size of each pixel in mm")
    height_step_mm: float = Field(default=0.0, description="Height quantization step. 0 for continuous height")
    face_up : bool = Field(default=True, description="Whether the lithophane is viewed from the top (face up) or bottom (face down)")
    intensity_min_height: float = Field(default=0.2, description="Minimum height for intensity layers")
    color_correction: ColorCorrection = Field(default=ColorCorrection.LINEAR, description="Color correction method")
    luminance_config: LuminanceConfig = Field(default_factory=LuminanceConfig)
    filament_library: Dict[LayerType, Filament] = Field(default_factory=lambda: {})
    include_clear_filler: bool = Field(default=True, description="Whether to add a clear filler layer before the white intensity layer")

class StlCollection(BaseModel):
    meshes: Dict[str, Mesh]
    
    class Config:
        arbitrary_types_allowed = True
    
    def __getitem__(self, key: str) -> Mesh:
        return self.meshes[key]
    
    def __iter__(self):
        return iter(self.meshes.values())
    
    def items(self):
        return self.meshes.items()

    def save_to_folder(self, output_dir):
        for k,v in self.meshes.items():
            v.save(os.path.join(output_dir, f"{k}.stl"))
