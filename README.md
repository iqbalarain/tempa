# Watermark Removal Tool

This tool uses AI-powered inpainting to remove watermarks from images.

## Installation

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. The script will automatically download the LaMa inpainting model on first run.

## Usage

1. Place your watermarked images in the `input_images` folder.
2. Run the script:
   ```bash
   python watermark_remove.py
   ```
3. The script will attempt to automatically detect the watermark location.
4. If detection fails, you will be prompted to enter watermark coordinates manually.
5. Processed images will be saved to the `processed_images` folder.

## Advanced Options

You can modify these variables in the script:
- `input_directory`: Folder containing watermarked images
- `output_directory`: Where processed images will be saved
- `threshold`: Sensitivity for watermark detection (default: 30)

## Fallback Method

If the LaMa model fails to load or process, the script will fall back to using OpenCV's inpainting methods.