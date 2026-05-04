#!/usr/bin/python3

from ImageAnalyzer import ImageAnalyzer
from Models import ColorCorrection, LuminanceConfig
from filaments import FilamentLibrary
from to_stl import LayerType, to_stl_cym, StlConfig
import os
import cv2
import argparse
from pathlib import Path


# Set up command line arguments
parser = argparse.ArgumentParser(description='Process an image into CMYK 3D printable layers')
parser.add_argument('--show-images', action='store_true', default=False,
                   help='Display the original and processed images')
parser.add_argument('--input', '-i', default='examples/girl-with-pearl-earings.png',
                   help='Input image file path')
parser.add_argument('--output-image', '-o', default='',
                   help='Output pixelated image file path')
parser.add_argument('--width', '-w', type=float, default=50,
                   help='Desired width in mm')
parser.add_argument('--resolution', '-r', type=float, default=0.4,
                   help='Resolution in mm per pixel/block')
parser.add_argument('--stl-output', default='stl-output',
                   help='Output directory for STL files')
parser.add_argument('--face-up', action='store_true', default=False,
                   help='Mirror STLs for face-up viewing')
parser.add_argument('--no-clear', action='store_true', default=False,
                   help='Skip the clear filler layer to reduce total thickness')
parser.add_argument('--filament-label', default=None,
                   help='Select a complete CMYK filament set from filaments.yaml by label')
parser.add_argument('--cym-target-thickness', type=float, default=0.07,
                   help='Target thickness of the cyan layer in mm')
parser.add_argument('--white-target-thickness', type=float, default=0.16,
                   help='Target thickness of the white layer in mm')

args = parser.parse_args()

# Replace variables with command line arguments
show_images = args.show_images
ifile = args.input
ofile = args.output_image
desired_width_mm = args.width
resolution_mm = args.resolution
face_up = args.face_up

# Create output directory if it doesn't exist
stl_output_dir = args.stl_output
os.makedirs(stl_output_dir, exist_ok=True)

# Calculate number of blocks based on desired physical width and resolution
n_blocks = int(desired_width_mm / resolution_mm)

img = ImageAnalyzer(ifile)
img_info = img.get_image_info()
x_pixels = img_info.get('width')
y_pixels = img_info.get('height')

# Calculate block size in pixels
block_size = int(x_pixels / n_blocks)

# Calculate physical height maintaining aspect ratio
physical_height_mm = (y_pixels / x_pixels) * desired_width_mm

print(f"Image will be divided into {n_blocks}x{int(n_blocks * (y_pixels/x_pixels))} blocks")
print(f"Each block will be {resolution_mm}mm x {resolution_mm}mm")
print(f"Final dimensions will be {desired_width_mm}mm x {physical_height_mm:.1f}mm")
# Usage example
yaml_path = Path("filaments.yaml")
library = FilamentLibrary.from_yaml(yaml_path)
filament_library = library.get_filament_set(args.filament_label) if args.filament_label else {
    LayerType.CYAN: library.get_filament("bambu_cyan_pla"),     # RGB for Cyan
    LayerType.YELLOW: library.get_filament("bambu_yellow_pla"),   # RGB for Yellow
    LayerType.MAGENTA: library.get_filament("bambu_magenta_pla"),  # RGB for Magenta
    LayerType.WHITE: library.get_filament("bambu_white_pla"),      # RGB for White
}


img.pixelate(block_size)

if not ofile == '' and not ofile is None:
    img.save_processed_image(ofile)

# Display the original and processed images
if show_images:
    cv2.imshow('Original Image', cv2.cvtColor(img.original, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.imshow('Processed Image', cv2.cvtColor(img.pixelated, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Create STL configuration
stl_config = StlConfig(
    pixel_size=resolution_mm,
    base_height=0.2,
    intensity_min_height=0.2,
    height_step_mm=0.1,
    face_up=face_up,
    luminance_config = LuminanceConfig(
        cym_target_thickness=args.cym_target_thickness,
        white_target_thickness=args.white_target_thickness,
    ),
    color_correction=ColorCorrection.LUMINANCE,
    include_clear_filler=not args.no_clear,
    filament_library=filament_library
)
print(f"\nSTL Configuration:\n{stl_config.model_dump_json(indent=2)}")

if __name__ == "__main__":
    stl_collection = to_stl_cym(
        img,
        config=stl_config
    )

    stl_collection.save_to_folder(stl_output_dir)
