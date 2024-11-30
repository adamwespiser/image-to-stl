from ImageAnalyzer import ImageAnalyzer
import numpy as np
from stl.mesh import Mesh
from typing import Tuple
from Models import ColorCorrection, LayerType, StlConfig, StlCollection
from color_mixing import extract_and_invert_channels, extract_and_invert_channels_linear

def create_layer_mesh(height_map: np.ndarray,
                     height_step_mm: float,
                     pixel_size: float,
                     previous_heights: np.ndarray = None,
                     min_height: float = 0,
                     flat_top: bool = False,
                     ) -> Tuple[Mesh, np.ndarray]:
    y_pixels, x_pixels = height_map.shape
    
    # Vectorized height calculations
    previous_heights = np.zeros_like(height_map) if previous_heights is None else previous_heights
    max_height = np.max(previous_heights) + min_height if flat_top else 0
    
    if flat_top:
        z = np.full_like(height_map, max_height) - previous_heights
    else:
        z = height_map.copy()
        if height_step_mm > 0:
            z = np.round(z / height_step_mm) * height_step_mm
        z = np.maximum(z, min_height)
    
    next_heights = z + previous_heights
    
    x_coords, y_coords = np.meshgrid(np.arange(x_pixels), np.arange(y_pixels))
    
    vertices = np.zeros((y_pixels, x_pixels, 8, 3))
    
    # Bottom vertices
    vertices[:, :, 0] = np.stack([x_coords * pixel_size, y_coords * pixel_size, previous_heights], axis=-1)
    vertices[:, :, 1] = np.stack([(x_coords + 1) * pixel_size, y_coords * pixel_size, previous_heights], axis=-1)
    vertices[:, :, 2] = np.stack([(x_coords + 1) * pixel_size, (y_coords + 1) * pixel_size, previous_heights], axis=-1)
    vertices[:, :, 3] = np.stack([x_coords * pixel_size, (y_coords + 1) * pixel_size, previous_heights], axis=-1)
    
    # Top vertices
    vertices[:, :, 4] = np.stack([x_coords * pixel_size, y_coords * pixel_size, next_heights], axis=-1)
    vertices[:, :, 5] = np.stack([(x_coords + 1) * pixel_size, y_coords * pixel_size, next_heights], axis=-1)
    vertices[:, :, 6] = np.stack([(x_coords + 1) * pixel_size, (y_coords + 1) * pixel_size, next_heights], axis=-1)
    vertices[:, :, 7] = np.stack([x_coords * pixel_size, (y_coords + 1) * pixel_size, next_heights], axis=-1)
    
    vertices = vertices.reshape(-1, 3)
    
    # Faces
    pixel_indices = np.arange(y_pixels * x_pixels * 8).reshape(y_pixels, x_pixels, 8)
    base_indices = pixel_indices[:, :, 0].reshape(-1)
    face_template = np.array([
        [0, 2, 1], [0, 3, 2],  # bottom
        [4, 5, 6], [4, 6, 7],  # top
        [0, 1, 5], [0, 5, 4],  # front
        [2, 3, 7], [2, 7, 6],  # back
        [0, 4, 7], [0, 7, 3],  # left
        [1, 2, 6], [1, 6, 5]   # right
    ])
    
    # Create offset array for each pixel
    offsets = np.arange(0, len(base_indices) * 8, 8)[:, None, None]
    
    # Broadcasting to create all faces at once
    faces = (face_template[None, :, :] + offsets).reshape(-1, 3)
    
    faces = np.array(faces)
    
    # Create mesh and compute normals
    stl_mesh = Mesh(np.zeros(len(faces), dtype=Mesh.dtype))
    stl_mesh.vectors = vertices[faces]
    
    # Vectorized normal calculation
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    edge1 = v1 - v0
    edge2 = v2 - v0
    normals = np.cross(edge1, edge2)
    
    # Normalize non-zero normals
    norms = np.linalg.norm(normals, axis=1)
    mask = norms > 0
    normals[mask] = normals[mask] / norms[mask, np.newaxis]
    normals[~mask] = [0, 0, 1]
    
    stl_mesh.normals = normals
    
    return stl_mesh, next_heights

def create_base_plate(x_pixels: int, y_pixels: int, config: StlConfig) -> Mesh:
    height_map = np.full((y_pixels, x_pixels), config.base_height, dtype=float)
    
    base_mesh, _ = create_layer_mesh(
        height_map=height_map,
        height_step_mm=config.height_step_mm,
        pixel_size=config.pixel_size,
        previous_heights=np.zeros((y_pixels, x_pixels))
    )
    
    return base_mesh

def create_color_layer(height_map: np.ndarray, 
                      previous_heights: np.ndarray,
                      config: StlConfig,
                      layer_type: LayerType,
                      flat_top: bool = False) -> Tuple[Mesh, np.ndarray]:
    return create_layer_mesh(
        height_map=height_map,
        height_step_mm=config.height_step_mm,
        pixel_size=config.pixel_size,
        previous_heights=previous_heights,
        min_height=config.intensity_min_height if layer_type == LayerType.WHITE else 0,
        flat_top=flat_top
      
    )


def to_stl_cym(img: ImageAnalyzer, config: StlConfig = None) -> StlCollection:
    if config is None:
        config = StlConfig()
        
    if len(img.pixelated.shape) != 3 or img.pixelated.shape[2] != 3:
        raise ValueError("Image must have 3 channels (CYM)")

    intensity_channels = extract_and_invert_channels(img, config) if config.color_correction == ColorCorrection.LUMINANCE else extract_and_invert_channels_linear(img, config)
    y_pixels, x_pixels = img.pixelated.shape[:2]
    
    print("creating stl: white_base_mesh.stl")
    base_mesh = create_base_plate(x_pixels, y_pixels, config)
    base_heights = np.full((y_pixels, x_pixels), config.base_height, dtype=float)
    print("base_heights " + str(config.base_height))

    layers = {
        'cyan_mesh': (intensity_channels.c_channel, base_heights, LayerType.CYAN),
        'yellow_mesh': (intensity_channels.y_channel, None, LayerType.YELLOW),
        'magenta_mesh': (intensity_channels.m_channel, None, LayerType.MAGENTA),
        'clear_mesh': (intensity_channels.intensity_map, None, LayerType.CLEAR),
        'white_intensity_mesh': (intensity_channels.intensity_map, None, LayerType.WHITE)
    }
    
    previous_heights = base_heights
    meshes = {'white_base_mesh': base_mesh}
    
    for name, (height_map, _, layer_type) in layers.items():
        print("creating stl: " + name + ".stl")
        mesh, previous_heights = create_color_layer(
            height_map=height_map,
            previous_heights=previous_heights,
            config=config,
            layer_type=layer_type,
            flat_top=layer_type == LayerType.CLEAR,
        )
        meshes[name] = mesh
    
    return StlCollection(meshes=meshes)