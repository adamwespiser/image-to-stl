import numpy as np
from functools import lru_cache
from Models import Filament, LayerType, IntensityChannels, LuminanceConfig, StlConfig
from ImageAnalyzer import ImageAnalyzer
from typing import Dict, Tuple
from scipy.optimize import minimize


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    # Remove the hash symbol if present
    hex_color = hex_color.lstrip('#')
    
    # Check if the hex code is valid
    if len(hex_color) not in (3, 6):
        raise ValueError("Invalid hex color code. Must be 3 or 6 characters long.")
    
    # If it's a 3-digit hex code, expand it to 6 digits
    if len(hex_color) == 3:
        hex_color = ''.join([c * 2 for c in hex_color])
    
    # Convert hex to RGB
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    
    return (max(r,0), max(g,0), max(b,0))

def calculate_color_thicknesses(
    target_rgb: np.ndarray,
    filaments: Dict[LayerType, Filament],
    luminance_config: LuminanceConfig,
    beer_lamport: bool = False,
    true_color: bool = True,
) -> Tuple[float, float, float, float]:
    return calculate_color_thicknesses_cached(
        target_rgb[0],
        target_rgb[1],
        target_rgb[2],
        filaments[LayerType.CYAN].hex_value,
        filaments[LayerType.YELLOW].hex_value,
        filaments[LayerType.MAGENTA].hex_value,
        filaments[LayerType.WHITE].hex_value,
        filaments[LayerType.CYAN].transmission_distance,
        filaments[LayerType.YELLOW].transmission_distance,
        filaments[LayerType.MAGENTA].transmission_distance,
        filaments[LayerType.WHITE].transmission_distance,
        luminance_config.cym_target_thickness,
        luminance_config.white_target_thickness,
        beer_lamport,
        true_color
    )



@lru_cache(maxsize=1200)
def calculate_color_thicknesses_cached(
    target_r,
    target_g,
    target_b,
    c_filament_hex: str,
    y_filament_hex: str,
    m_filament_hex: str,
    k_filament_hex: str,
    c_filament_td: float,
    y_filament_td: float,
    m_filament_td: float,
    k_filament_td: float,
    cym_target_thickness: float,
    white_target_thickness: float,
    beer_lamport: bool = False,
    true_color: bool = True,
) -> Tuple[float, float, float, float]:
    """
    Calculate thicknesses for CMY layers using CMYK conversion and filament properties.
    Returns (cyan, magenta, yellow, black) thicknesses.
    """
    # Convert RGB [0-255] to [0-1] scale
    target_rgb = np.array([target_r, target_g, target_b], dtype=float)
    rgb = target_rgb / 255.0
    epsilon = 1e-5  # Prevent log(0)

    if true_color:
        cyan_rgb = np.array(hex_to_rgb(c_filament_hex)) / 255.0
        magenta_rgb = np.array(hex_to_rgb(m_filament_hex)) / 255.0
        yellow_rgb = np.array(hex_to_rgb(y_filament_hex)) / 255.0
        white_rgb = np.array(hex_to_rgb(k_filament_hex)) / 255.0

        def calculate_achieved_rgb(cmy_amounts, k_amount):
            """Calculate resulting RGB values for given CMY and K amounts using filter model"""
            # Ensure amounts are positive and normalized
            amounts = np.clip(cmy_amounts, 0, 1)
            k_amount = np.clip(k_amount, 0, 1)
            
            # Start with white base
            achieved = white_rgb.copy()
            
            # Apply K (black) filter first
            achieved *= (1 - k_amount)
            
            # Apply each color filter's contribution
            if amounts[0] > 0:  # Cyan
                achieved *= (1 - amounts[0] * (1 - cyan_rgb))
            if amounts[1] > 0:  # Magenta
                achieved *= (1 - amounts[1] * (1 - magenta_rgb))
            if amounts[2] > 0:  # Yellow
                achieved *= (1 - amounts[2] * (1 - yellow_rgb))
            
            return achieved * 255.0
        
        def color_error(cmyk_amounts):
            # Split into CMY and K components
            cmy_amounts = cmyk_amounts[:3]
            k_amount = cmyk_amounts[3]
            
            achieved = calculate_achieved_rgb(cmy_amounts, k_amount)
            
            return np.mean(((achieved - target_rgb) / 255.0) ** 2)
        
        # Update bounds to include K
        bounds = [(0, 1) for _ in range(4)]  # Now CMYK instead of just CMY

        # Use a few deterministic starts to reduce local-minimum and tie issues.
        k0 = 1 - np.max(rgb)
        denom = 1 - k0 + epsilon
        c0 = (1 - rgb[0] - k0) / denom
        m0 = (1 - rgb[1] - k0) / denom
        y0 = (1 - rgb[2] - k0) / denom
        initial_guesses = [
            np.clip(np.array([c0, m0, y0, k0]), 0, 1),
            np.array([0.5, 0.5, 0.5, 0.5]),
            np.array([0.0, 0.0, 0.0, k0]),
            np.array([0.0, 0.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 1.0, 0.0]),
        ]

        results = [
            minimize(
                color_error,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={
                    'ftol': 1e-10,
                    'maxiter': 300
                }
            )
            for x0 in initial_guesses
        ]

        best_color_error = min(color_error(result.x) for result in results)
        best_results = [
            result for result in results
            if color_error(result.x) <= best_color_error + 1e-10
        ]

        def total_printed_thickness(cmyk_amounts):
            c, m, y, k = cmyk_amounts
            return (
                c * cym_target_thickness * c_filament_td
                + m * cym_target_thickness * m_filament_td
                + y * cym_target_thickness * y_filament_td
                + k * white_target_thickness * k_filament_td
            )

        result = min(best_results, key=lambda candidate: total_printed_thickness(candidate.x))

        # Extract results
        cmy_thicknesses = result.x[:3]
        k_thickness = result.x[3]
        
        # Debug prints
        print("Target RGB:", rgb * 255.0)
        print("Achieved RGB:", calculate_achieved_rgb(cmy_thicknesses, k_thickness))
        print("Error:", color_error(result.x))
        print("Solution:", cmy_thicknesses)
        
        c = cmy_thicknesses[0]
        m = cmy_thicknesses[1]
        y = cmy_thicknesses[2]
        k = k_thickness
    else:
        # Convert RGB to CMYK
        k = 1 - np.max(rgb)
        c = (1 - rgb[0] - k) / (1 - k + epsilon)  # Add small epsilon to prevent division by zero
        m = (1 - rgb[1] - k) / (1 - k + epsilon)
        y = (1 - rgb[2] - k) / (1 - k + epsilon)
    
    cyan_thickness = 0
    magenta_thickness = 0
    yellow_thickness = 0
    white_thickness = 0
    
    if beer_lamport: 
        # Apply Beer-Lambert law: T = e^(-α * l), where:
        # T is transmission (we want to solve for l - thickness)
        # α is absorption coefficient (can be derived from filament properties)
        # Solve for thickness: l = -ln(T) / α

        # Calculate thicknesses using Beer-Lambert law
        cyan_thickness = (-np.log(max(1 - c, epsilon))  * c_filament_td) * cym_target_thickness
        magenta_thickness = (-np.log(max(1 - m, epsilon)) * m_filament_td) * cym_target_thickness
        yellow_thickness = (-np.log(max(1 - y, epsilon)) * y_filament_td) * cym_target_thickness
        white_thickness = (-np.log(max(1 - k, epsilon)) * k_filament_td) * white_target_thickness
        
    else:
        # Scale thicknesses by filament properties and transmission distance
        cyan_thickness = c * cym_target_thickness * c_filament_td
        magenta_thickness = m * cym_target_thickness * m_filament_td
        yellow_thickness = y * cym_target_thickness * y_filament_td
        white_thickness = k * white_target_thickness * k_filament_td
    
    # Clip to physical constraints
    cyan_thickness = np.clip(cyan_thickness, 0, c_filament_td)
    magenta_thickness = np.clip(magenta_thickness, 0, m_filament_td)
    yellow_thickness = np.clip(yellow_thickness, 0, y_filament_td)
    white_thickness = np.clip(white_thickness, 0, k_filament_td)
    
    return cyan_thickness, magenta_thickness, yellow_thickness, white_thickness

def calculate_exact_thicknesses(
    target_rgb: np.ndarray,
    filaments: Dict[LayerType, Filament],
    luminance_config: LuminanceConfig
) -> Tuple[float, float, float, float]:
    """Calculate all layer thicknesses"""
    c, m, y, k = calculate_color_thicknesses(target_rgb, filaments, luminance_config)
    return c, m, y, k

def extract_and_invert_channels(img: ImageAnalyzer, config: StlConfig) -> IntensityChannels:
    """Process entire image"""
    shape = img.pixelated.shape[:2]
    c_channel = np.zeros(shape)
    y_channel = np.zeros(shape)
    m_channel = np.zeros(shape)
    w_channel = np.zeros(shape)
    
    # Process each pixel
    for i in range(shape[0]):
        for j in range(shape[1]):
            c, m, y, w = calculate_exact_thicknesses(
                img.pixelated[i,j], 
                config.filament_library, 
                config.luminance_config
            )
            c_channel[i,j] = c
            m_channel[i,j] = m
            y_channel[i,j] = y
            w_channel[i,j] = w
    
    return IntensityChannels(
        c_channel=c_channel,
        y_channel=y_channel,
        m_channel=m_channel,
        intensity_map=w_channel
    )

def normalize_thickness_linear(intensity: np.ndarray, max_distance: float, min_thickness: float) -> np.ndarray:
    """
    Convert intensity values (0-255) to thickness values
    Uses linear mapping where:
    - intensity 0 (black) = max_distance (fully blocks light)
    - intensity 255 (white) = min_thickness (maximum light transmission)
    """
    # Normalize intensity to [0, 1]
    normalized = intensity / 255.0
    
    # Linear interpolation between min_thickness and max_distance
    thickness = (1 - normalized) * (max_distance - min_thickness) + min_thickness
    
    return thickness

def extract_and_invert_channels_linear(img: ImageAnalyzer, config: StlConfig) -> IntensityChannels:
    c_channel = normalize_thickness_linear(img.pixelated[:, :, 0], config.filament_library[LayerType.CYAN].transmission_distance, 0)
    y_channel = normalize_thickness_linear(img.pixelated[:, :, 1], config.filament_library[LayerType.YELLOW].transmission_distance, 0) 
    m_channel = normalize_thickness_linear(img.pixelated[:, :, 2], config.filament_library[LayerType.MAGENTA].transmission_distance, 0)
    
    # For intensity map, use average of RGB then scale to KEY layer heights
    avg_pixels = (img.pixelated[:, :, 0] + img.pixelated[:, :, 1] + img.pixelated[:, :, 2]) / 3.0
    intensity_map = normalize_thickness_linear(avg_pixels, config.filament_library[LayerType.WHITE].transmission_distance, config.intensity_min_height)
    
    return IntensityChannels(
        c_channel=c_channel,
        y_channel=y_channel,
        m_channel=m_channel,
        intensity_map=intensity_map
    ) 
