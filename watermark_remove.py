import cv2
import numpy as np
import os
import glob
from tqdm import tqdm

def remove_watermark_with_inpainting(image_path, output_dir, watermark_region):
    """
    Remove watermark from an image using AI-based inpainting.
    
    Args:
        image_path: Path to the input image
        output_dir: Directory to save processed images
        watermark_region: Tuple of (left, top, right, bottom) coordinates of the watermark
    """
    try:
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not read image: {image_path}")
            return False
            
        # Create a mask for the watermark region (white where the watermark is)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        left, top, right, bottom = watermark_region
        mask[top:bottom, left:right] = 255
        
        # Apply inpainting
        # INPAINT_TELEA is for Navier-Stokes based inpainting
        # INPAINT_NS is for Fast Marching Method based inpainting
        radius = 3  # Radius of a circular neighborhood of each point
        result = cv2.inpaint(img, mask, radius, cv2.INPAINT_NS)
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the processed image
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, result)
        
        return True
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False

def batch_process_images(input_dir, output_dir, watermark_region):
    """
    Process all JPG images in the input directory and save to output directory.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save processed images
        watermark_region: Tuple of (left, top, right, bottom) coordinates of the watermark
    """
    # Get all JPG files in the directory
    image_paths = glob.glob(os.path.join(input_dir, "*.jpg")) + glob.glob(os.path.join(input_dir, "*.jpeg"))
    
    total = len(image_paths)
    successful = 0
    
    print(f"Found {total} JPG images to process.")
    
    # Use tqdm for a progress bar
    for image_path in tqdm(image_paths, desc="Processing images"):
        if remove_watermark_with_inpainting(image_path, output_dir, watermark_region):
            successful += 1
    
    print(f"Processing complete. Successfully processed {successful}/{total} images.")

def auto_detect_watermark(sample_image_path, threshold=30):
    """
    Attempt to automatically detect a watermark region in a sample image.
    This is a basic implementation and may need fine-tuning.
    
    Args:
        sample_image_path: Path to a sample image with watermark
        threshold: Threshold for edge detection
    
    Returns:
        Tuple of (left, top, right, bottom) coordinates of the detected watermark region
    """
    # Read the image
    img = cv2.imread(sample_image_path)
    if img is None:
        print(f"Could not read image: {sample_image_path}")
        return None
        
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply edge detection
    edges = cv2.Canny(gray, threshold, threshold*2)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour - this might be the watermark
    if contours:
        # Sort contours by area
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Get bounding rectangle of the largest contour
        x, y, w, h = cv2.boundingRect(contours[0])
        
        # Return the coordinates
        return (x, y, x+w, y+h)
    
    print("Could not detect watermark automatically.")
    return None

if __name__ == "__main__":
    # Configuration
    input_directory = "input_images"  # Change this to your images folder
    output_directory = "processed_images"  # Output folder
    
    # Sample image for watermark detection
    sample_image = os.path.join(input_directory, os.listdir(input_directory)[0])
    
    # Try to detect watermark automatically
    print("Attempting to detect watermark automatically...")
    watermark_region = auto_detect_watermark(sample_image)
    
    if watermark_region:
        print(f"Detected watermark region: {watermark_region}")
    else:
        # Fall back to manual coordinates if detection fails
        print("Using default watermark coordinates. Please adjust if needed.")
        watermark_region = (50, 50, 200, 100)  # Example coordinates
    
    # Process all images
    batch_process_images(input_directory, output_directory, watermark_region)