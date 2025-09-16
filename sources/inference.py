import os
import glob
import shutil
from pathlib import Path
from ultralytics import YOLO

def convert_to_openvino_if_needed(model_path, openvino_dir='openvino_model'):
    """
    Convert YOLO model to OpenVINO format if not already converted
    """
    # Create OpenVINO directory if it doesn't exist
    os.makedirs(openvino_dir, exist_ok=True)
    
    # Check if OpenVINO model already exists (look for the directory structure)
    openvino_xml = os.path.join(openvino_dir, 'best.xml')
    openvino_bin = os.path.join(openvino_dir, 'best.bin')
    
    if not os.path.exists(openvino_xml) or not os.path.exists(openvino_bin):
        print("Converting model to OpenVINO format...")
        try:
            # Load the original model
            model = YOLO(model_path)
            
            # Export to OpenVINO format - let Ultralytics handle the directory creation
            # Don't move files, use the directory created by Ultralytics
            export_path = model.export(format='openvino', imgsz=640, half=False, device='cpu')
            
            if export_path and os.path.exists(export_path):
                print(f"OpenVINO model saved to: {export_path}")
                return export_path  # Return the path created by Ultralytics
            else:
                print("Warning: OpenVINO export may have failed!")
                return model_path
            
        except Exception as e:
            print(f"Error converting to OpenVINO: {e}")
            return model_path
    
    return openvino_dir

def find_image_files(directory='.'):
    """
    Find image files in the specified directory, excluding those starting with 'result'
    """
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
    image_files = []
    
    for extension in image_extensions:
        # Get all files with this extension
        all_files = glob.glob(os.path.join(directory, extension))
        
        # Filter out files that start with 'result' (case insensitive)
        for file in all_files:
            filename = os.path.basename(file)
            if not filename.lower().startswith('result'):
                image_files.append(file)
    
    return sorted(image_files)

def main():
    # Model paths
    original_model_path = 'best.pt'
    
    # Check if original model exists
    if not os.path.exists(original_model_path):
        print(f"Error: Model file '{original_model_path}' not found!")
        return
    
    # Convert to OpenVINO if needed and get the appropriate model path
    model_path = convert_to_openvino_if_needed(original_model_path)
    
    # Load the appropriate model
    try:
        print(f"Loading model from: {model_path}")
        
        # Try different loading approaches for OpenVINO
        if 'openvino' in model_path.lower() or os.path.isdir(model_path):
            # For OpenVINO, try different loading methods
            try:
                # Method 1: Load as directory
                model = YOLO(model_path)
                print("Model loaded successfully as OpenVINO directory")
            except Exception as e:
                print(f"Error loading OpenVINO model as directory: {e}")
                # Fall back to PyTorch model
                print("Falling back to PyTorch model...")
                model = YOLO(original_model_path)
        else:
            # Load PyTorch model
            model = YOLO(model_path)
            print("PyTorch model loaded successfully")
            
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Falling back to PyTorch model...")
        model = YOLO(original_model_path)
    
    # Find image files in current directory, excluding result images
    image_files = find_image_files('.')
    
    if not image_files:
        print("No image files found in current directory (excluding result images)!")
        return
    
    print(f"Found {len(image_files)} image file(s) (excluding result images):")
    for img_file in image_files:
        print(f"  - {img_file}")
    
    # Process each image
    for image_path in image_files:
        print(f"\nProcessing image: {image_path}")
        
        try:
            # Perform inference
            results = model(image_path, imgsz=640, conf=0.5)
            
            # Generate output filename
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = f"result_{base_name}.jpg"
            
            # Save results
            for i, result in enumerate(results):
                result.save(filename=output_path)
                print(f"Result saved to: {output_path}")
                
                # Print detection information
                if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                    print(f"Detections: {len(result.boxes)} objects")
                    for j, box in enumerate(result.boxes):
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        class_name = result.names[cls] if hasattr(result, 'names') else str(cls)
                        print(f"  {j+1}. {class_name}: {conf:.3f} confidence")
                else:
                    print("No objects detected")
                    
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

if __name__ == "__main__":
    main()