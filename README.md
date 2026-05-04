# Color Lithophane Project

This project prepares color lithophanes for 3D printing by converting an input
image into separate STL parts for a multi-material CMY plus white/clear print.

Current assumptions:
* Simple CMY/white subtractive color model
* Filament parameters from `filaments.yaml`
* Resolution is configured as millimeters per image block

## Examples

![Example 1](./examples/example_printed.png)

## Setup

This project requires Python 3.13 or newer.

```bash
uv venv
uv sync
source .venv/bin/activate
```

## Running the Project

Generate STLs from the default example image:

```bash
python src/main.py
```

Generate from a custom image:

```bash
python src/main.py \
  --input examples/sample.png \
  --width 50 \
  --resolution 0.4 \
  --filament-label bambu \
  --stl-output stl-output
```

The script writes six STL parts:
* `white_base_mesh.stl`
* `cyan_mesh.stl`
* `yellow_mesh.stl`
* `magenta_mesh.stl`
* `clear_mesh.stl`
* `white_intensity_mesh.stl`

Use `--no-clear` to skip `clear_mesh.stl` and reduce total print thickness.

## Options for generation

```bash
$ python src/main.py --help
usage: main.py [-h] [--show-images] [--input INPUT]
               [--output-image OUTPUT_IMAGE] [--width WIDTH]
               [--resolution RESOLUTION] [--stl-output STL_OUTPUT] [--face-up]
               [--no-clear]
               [--filament-label FILAMENT_LABEL]
               [--cym-target-thickness CYM_TARGET_THICKNESS]
               [--white-target-thickness WHITE_TARGET_THICKNESS]

Process an image into CMYK 3D printable layers

options:
  -h, --help            show this help message and exit
  --show-images         Display the original and processed images
  --input, -i INPUT     Input image file path
  --output-image, -o OUTPUT_IMAGE
                        Output pixelated image file path
  --width, -w WIDTH     Desired width in mm
  --resolution, -r RESOLUTION
                        Resolution in mm per pixel/block
  --stl-output STL_OUTPUT
                        Output directory for STL files
  --face-up             Mirror STLs for face-up viewing
  --no-clear            Skip the clear filler layer to reduce total thickness
  --filament-label FILAMENT_LABEL
                        Select a complete CMYK filament set from filaments.yaml by label
  --cym-target-thickness CYM_TARGET_THICKNESS
                        Target thickness of the cyan layer in mm
  --white-target-thickness WHITE_TARGET_THICKNESS
                        Target thickness of the white layer in mm
```

## Bambu Studio Setup

1. Launch Bambu Studio
2. File → Import → Select all six generated STL files
3. Click "Yes" when prompted to load all files as a single object with multiple parts
4. Assign each part to its matching filament

If generated with `--no-clear`, import the five generated STL files instead.

## Validation Print

Generate a measured CMYK validation chart for camera-based calibration:

```bash
python src/generate_validation_print.py \
  --width 60 \
  --resolution 0.4 \
  --filament-label bambu \
  --output stl-output/validation-print
```

The validation generator writes STL parts, a `validation_chart_metadata.json`
file with the known CMYK values for each patch, and a
`validation_chart_preview.png` guide image. It uses a 4mm border and skips the
clear filler layer by default. Add `--with-clear` if you want the validation
piece to match a clear-filler print stack.

Filament labels are defined with the optional list-valued `label` field in
`filaments.yaml`. A selected label must provide one cyan, yellow, magenta, and
white/intensity filament.

## Color Set Ranking

Compare complete labeled filament sets before printing:

```bash
python src/color-max.py
```

The utility scores each complete CMY+white label from `filaments.yaml` using:
* closeness to ideal cyan, magenta, yellow, and white RGB values
* simulated RGB mixing error across a coarse target color grid
* transmission-distance headroom relative to the configured height step

Use `--label bambu` to score one set, or `--json` for machine-readable output.
Use `--mix-and-match` to search the best cyan/magenta/yellow/white combination
across all candidate filaments instead of only scoring existing labels.

### Print Settings
Configure the following settings for optimal results:

- Nozzle diameter: 0.4mm
- First layer height: 0.2mm
- Subsequent layer heights: 0.1mm

## Project TODO

### Mixing algorithm
- [x] Implement transmission distance model
- [ ] Implement more optimal subtractive color mixing model

### Usability
- [x] Add yaml config for filament parameters
- [ ] Replace resolution with nozzle sizes (0.2mm and 0.4mm)
- [ ] Implement error handling for invalid input images
- [ ] Add progress bar for long operations
- [ ] Optimize memory usage for large images


### Communication
- [ ] Better documentation for parameters
- [ ] Create library of commonly used filaments
- [ ] Create example gallery with sample prints
- [x] Add calibration pattern generation
- [ ] Create troubleshooting guide

### General
- [ ] Add test suite for core functionality
- [ ] Add support for web server hosting
- [ ] Refactor/clean-up code
