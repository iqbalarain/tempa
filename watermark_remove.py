import os
import glob
import torch
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import requests
import zipfile
import io
import sys

def download_lama_model():
    """Download the pre-trained LaMa model if not already available"""
    model_dir = "lama_model"
    model_path = os.path.join(model_dir, "big-lama.pt")
    
    if os.path.exists(model_path):
        print("LaMa model already downloaded.")
        return model_path
    
    print("Downloading LaMa model... This may take a few minutes.")
    os.makedirs(model_dir, exist_ok=True)
    
    # URL for the LaMa model
    url = "https://github.com/advimman/lama/releases/download/v1.0/big-lama.pt"
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        
        with open(model_path, 'wb') as f:
            for data in tqdm(response.iter_content(block_size), total=total_size//block_size, 
                            desc="Downloading LaMa model"):
                f.write(data)
        
        # Verify file size as a basic integrity check
        if os.path.getsize(model_path) < 1000000:  # File should be several MB
            print("Downloaded file appears to be incomplete or corrupted")
            return None
            
        print(f"LaMa model downloaded to {model_path}")
        return model_path
    except Exception as e:
        print(f"Error downloading model: {e}")
        sys.exit(1)

def setup_lama_model():
    """Setup and return the LaMa model"""
    model_path = download_lama_model()
    
    if model_path is None:
        return None, None
        
    try:
        # Import here to avoid dependency issues if not using this function
        import torch
        
        # Load the model architecture - with more robust error handling
        try:
            model = torch.hub.load("saic-mdal/lama:main", "lama", pretrained=False, source='github')
        except Exception as e:
            print(f"Error loading model architecture: {e}")
            print("Attempting alternative loading method...")
            try:
                # Alternative loading method if the first one fails
                from torch.hub import load_state_dict_from_url
                # This would need a proper implementation based on model architecture
                print("Alternative loading method not fully implemented")
                return None, None
            except Exception:
                return None, None
        
        # Load the weights
        try:
            checkpoint = torch.load(model_path, map_location="cpu")
            model.load_state_dict(checkpoint['state_dict'])
        except Exception as e:
            print(f"Error loading model weights: {e}")
            return None, None
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()
        
        print(f"Model loaded successfully and running on {device}")
        return model, device
    except Exception as e:
        print(f"Error setting up LaMa model: {e}")
        print("Falling back to OpenCV inpainting method.")
        return None, None

def remove_watermark_with_lama(image_path, output_dir, watermark_region, model=None, device=None):
    """
    Remove watermark from an image using the LaMa inpainting model.
    Falls back to OpenCV if LaMa is not available.
    
    Args:
        image_path: Path to the input image
        output_dir: Directory to save processed images
        watermark_region: Tuple of (left, top, right, bottom) coordinates of the watermark
        model: Pre-loaded LaMa model
        device: Device to run inference on
    """
    try:
        # Add validation for image file
        if not os.path.exists(image_path):
            print(f"File not found: {image_path}")
            return False
            
        # Read the image with PIL for better color handling
        try:
            pil_img = Image.open(image_path)
            img = np.array(pil_img)
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            return False
            
        # Validate image dimensions
        if img.shape[0] < 10 or img.shape[1] < 10:
            print(f"Image too small: {image_path}")
            return False
        
        # Convert RGB to BGR for OpenCV compatibility if needed
        if len(img.shape) == 3 and img.shape[2] == 3:
            img_cv = img[:, :, ::-1].copy()  # RGB to BGR
        else:
            img_cv = img.copy()
        
        # Create a mask for the watermark region
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        left, top, right, bottom = watermark_region
        mask[top:bottom, left:right] = 255
        
        if model is not None and device is not None:
            # Process with LaMa model
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            mask_tensor = torch.from_numpy(mask).unsqueeze(0).float() / 255.0
            
            # Add batch dimension
            img_tensor = img_tensor.unsqueeze(0).to(device)
            mask_tensor = mask_tensor.unsqueeze(0).to(device)
            
            with torch.no_grad():
                result_tensor = model(img_tensor, mask_tensor)
                
            # Convert back to numpy and scale to 0-255
            result = (result_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        else:
            # Fall back to OpenCV inpainting
            radius = 3
            result = cv2.inpaint(img_cv, mask, radius, cv2.INPAINT_NS)
            
            # Convert back to RGB if needed
            if len(result.shape) == 3 and result.shape[2] == 3:
                result = result[:, :, ::-1]  # BGR to RGB
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the processed image
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, filename)
        
        # Save with PIL for better quality
        Image.fromarray(result).save(output_path, quality=95)
        
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
    # Setup LaMa model if possible
    try:
        model, device = setup_lama_model()
    except Exception as e:
        print(f"Error setting up LaMa model: {e}")
        print("Will use OpenCV inpainting instead.")
        model, device = None, None
    
    # Add directory validation
    if not os.path.exists(input_dir):
        print(f"Input directory does not exist: {input_dir}")
        return
    
    # Get all JPG files in the directory
    image_paths = glob.glob(os.path.join(input_dir, "*.jpg")) + glob.glob(os.path.join(input_dir, "*.jpeg"))
    
    total = len(image_paths)
    successful = 0
    
    print(f"Found {total} JPG images to process.")
    
    # Process images
    for image_path in tqdm(image_paths, desc="Processing images"):
        if remove_watermark_with_lama(image_path, output_dir, watermark_region, model, device):
            successful += 1
    
    print(f"Processing complete. Successfully processed {successful}/{total} images.")

def refine_watermark_detection(input_dir, n_samples=5, threshold=30):
    """
    Refine watermark detection by analyzing multiple sample images.
    
    Args:
        input_dir: Directory containing input images
        n_samples: Number of sample images to analyze
        threshold: Threshold for edge detection
    
    Returns:
        Tuple of (left, top, right, bottom) coordinates of the detected watermark region
    """
    print("Analyzing multiple images to accurately detect watermark...")
    
    # Get image paths
    image_paths = glob.glob(os.path.join(input_dir, "*.jpg")) + glob.glob(os.path.join(input_dir, "*.jpeg"))
    
    if not image_paths:
        print("No images found in the directory.")
        return None
    
    # Take a sample of images
    import random
    sample_paths = random.sample(image_paths, min(n_samples, len(image_paths)))
    
    regions = []
    for path in sample_paths:
        # Read the image
        img = cv2.imread(path)
        if img is None:
            continue
            
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, threshold, threshold*2)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Sort contours by area, take top 5
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                # Filter by reasonable size for a watermark (adjust as needed)
                if 10 < w < img.shape[1]//2 and 10 < h < img.shape[0]//2:
                    regions.append((x, y, x+w, y+h))
    
    if not regions:
        print("Could not detect any suitable watermark regions.")
        return None
    
    # Find the most common region (with some tolerance)
    def are_similar(r1, r2, tolerance=20):
        return (abs(r1[0] - r2[0]) < tolerance and
                abs(r1[1] - r2[1]) < tolerance and
                abs(r1[2] - r2[2]) < tolerance and
                abs(r1[3] - r2[3]) < tolerance)
    
    best_region = None
    max_count = 0
    
    for i, region in enumerate(regions):
        count = sum(1 for r in regions if are_similar(region, r))
        if count > max_count:
            max_count = count
            best_region = region
    
    if best_region:
        print(f"Detected watermark region: {best_region}")
        return best_region
    
    print("Could not reliably detect watermark region.")
    return None

if __name__ == "__main__":
    # Configuration
    input_directory = "input_images"  # Change this to your images folder
    output_directory = "processed_images"  # Output folder
    
    # Add directory validation and creation
    if not os.path.exists(input_directory):
        os.makedirs(input_directory, exist_ok=True)
        print(f"Created input directory: {input_directory}")
        print("Please place your images in this directory and run the script again.")
        sys.exit(0)
    
    # Add image file count check
    image_paths = glob.glob(os.path.join(input_directory, "*.jpg")) + glob.glob(os.path.join(input_directory, "*.jpeg"))
    if not image_paths:
        print("No JPG images found in the input directory.")
        print("Please add some images and run the script again.")
        sys.exit(0)
    
    # Try to detect watermark automatically with refined approach
    print("Attempting to detect watermark automatically...")
    watermark_region = refine_watermark_detection(input_directory, n_samples=5)
    
    if not watermark_region:
        # Fall back to manual coordinates if detection fails
        print("Using default watermark coordinates. Please adjust if needed.")
        watermark_region = (50, 50, 200, 100)  # Example coordinates
        
        # Ask user to confirm or adjust
        print("\nPlease confirm the watermark coordinates:")
        try:
            left = int(input(f"Left position (default: {watermark_region[0]}): ") or watermark_region[0])
            top = int(input(f"Top position (default: {watermark_region[1]}): ") or watermark_region[1])
            right = int(input(f"Right position (default: {watermark_region[2]}): ") or watermark_region[2])
            bottom = int(input(f"Bottom position (default: {watermark_region[3]}): ") or watermark_region[3])
            watermark_region = (left, top, right, bottom)
        except ValueError:
            print("Invalid input. Using default values.")
    
    # Process all images
    batch_process_images(input_directory, output_directory, watermark_region)