import cv2
import os
import glob
from tqdm import tqdm

# Path to the DAiSEE DataSet directory (contains Train, Test, Validation)
DATASET_DIR = "dataset/DAiSEE/DataSet"
# Number of frames to extract per video (Setting to 5 is sufficient for the repo's random.sample(3))
NUM_FRAMES_PER_VIDEO = 5

def extract_frames_from_video(video_path, num_frames):
    video_dir = os.path.dirname(video_path)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Fallback if total frame count cannot be read
    if total_frames <= 0:
        total_frames = 300 # Default DAiSEE video length is 10s at 30fps
        
   # Calculate evenly spaced frame indices to extract
    frame_indices = [int(i * (total_frames - 1) / (num_frames - 1)) for i in range(num_frames)]
    
    for count, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = cap.read()
        if success:
            frame_name = f"frame_{count}.jpg"
            frame_path = os.path.join(video_dir, frame_name)
            cv2.imwrite(frame_path, frame)
            
    cap.release()
    return True

def process_dataset():
    # Scan all .avi or .mp4 files inside the DataSet directory
    video_files = glob.glob(f"{DATASET_DIR}/*/*/*/*.avi") + glob.glob(f"{DATASET_DIR}/*/*/*/*.mp4")
    
    if not video_files:
        print(f"No videos found in {DATASET_DIR}. Please check the directory structure!")
        return

    print(f"Found {len(video_files)} videos. Extracting {NUM_FRAMES_PER_VIDEO} frames per video...")
    
    # Progress bar with tqdm
    for video_path in tqdm(video_files, desc="Extracting frames"):
        extract_frames_from_video(video_path, NUM_FRAMES_PER_VIDEO)
        
    print("Extraction complete! You can now run daisee_data_preprocessing.py")

if __name__ == "__main__":
    process_dataset()