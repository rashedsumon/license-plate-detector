import os
import glob
import kagglehub

def download_yolo_weights():
    """
    Downloads the license plate YOLO weights from Kaggle Hub 
    and returns the local file path to the .pt model file.
    """
    print("Checking/Downloading dataset from Kaggle...")
    # This downloads the dataset or returns the path if it already exists
    download_path = kagglehub.dataset_download("achrafkhazri/yolo-weights-for-licence-plate-detector")
    
    # Search for the PyTorch weights file (.pt) inside the downloaded directory
    pt_files = glob.glob(os.path.join(download_path, "**", "*.pt"), recursive=True)
    
    if not pt_files:
        raise FileNotFoundError(f"No YOLO .pt weight files found in {download_path}")
    
    # Return the path to the first detected weights file
    return pt_files[0]

if __name__ == "__main__":
    # Test script locally
    weight_path = download_yolo_weights()
    print(f"Success! Weights located at: {weight_path}")