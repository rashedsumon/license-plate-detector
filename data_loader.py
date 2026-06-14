import os
import glob
import kagglehub

def download_yolo_weights():
    """
    Downloads the license plate YOLO weights from Kaggle Hub 
    and returns paths to the .weights and .cfg files.
    """
    print("Checking/Downloading dataset from Kaggle...")
    download_path = kagglehub.dataset_download("achrafkhazri/yolo-weights-for-licence-plate-detector")
    
    # The dataset contains '.weights' and '.cfg' files instead of '.pt'
    weights_files = glob.glob(os.path.join(download_path, "**", "*.weights"), recursive=True)
    cfg_files = glob.glob(os.path.join(download_path, "**", "*.cfg"), recursive=True)
    
    if not weights_files or not cfg_files:
        raise FileNotFoundError(
            f"Could not find the expected darknet files (.weights or .cfg) in {download_path}. "
            f"Found files: {os.listdir(download_path)}"
        )
    
    # Return both file paths needed by Darknet YOLO frameworks
    return weights_files[0], cfg_files[0]

if __name__ == "__main__":
    w_path, c_path = download_yolo_weights()
    print(f"Success! Weights: {w_path}\nConfig: {c_path}")