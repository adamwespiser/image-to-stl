#!/usr/bin/python3

from ImageAnalyzer import ImageAnalyzer
from to_stl import to_stl_cym
import os
import cv2
import argparse


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

args = parser.parse_args()

# Replace variables with command line arguments
show_images = args.show_images
ifile = args.input
ofile = args.output_image
desired_width_mm = args.width
resolution_mm = args.resolution

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

stl_collection = to_stl_cym(
    img,
    pixel_size=resolution_mm,
    base_height=0.2,
    max_height=0.2,
    intensity_height=1,
    height_step_mm=0.1
)

stl_collection.save_to_folder(stl_output_dir)