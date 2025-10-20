import os
import shutil
import glob
import random
import evaluate as eval 
from PIL import Image 
import time

TARGET_SIZE = (256, 256) 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STAGE_DIR = os.path.join(BASE_DIR, "stage")
STUDY_DIR = os.path.join(BASE_DIR, "splitDataset", "study")
MODEL_PATH = "models/arshiv_2.pth/state_dict.pth"

FEEDBACK_DIR = os.path.join(BASE_DIR, "feedback")
ACCEPTED_DIR = os.path.join(FEEDBACK_DIR, "accepted")
REJECTED_DIR = os.path.join(FEEDBACK_DIR, "rejected")


def _resize_image(source_filepath: str, dest_path: str):
    try:
        img = Image.open(source_filepath).convert("RGB")
        w,h = img.size
        ratio = min(256/h, 256/w)
        new_width = int(ratio*w)
        new_height = int(ratio*h)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        bg = Image.new("RGB",(256, 256), (0,0,0))
        x = (256 - new_width)//2
        y = (256 - new_height)//2
        bg.paste(img, (x,y))
        bg.save(dest_path)

    except Exception as e:
        raise IOError(f"Could not resize or save image {source_filepath}: {e}")

def _init_dirs():
    os.makedirs(STAGE_DIR, exist_ok=True)
    os.makedirs(ACCEPTED_DIR, exist_ok=True)
    os.makedirs(REJECTED_DIR, exist_ok=True)

def clear_stage():
    files = glob.glob(os.path.join(STAGE_DIR, '*'))
    for f in files:
        try:
            os.remove(f)
        except OSError as e:
            print(f"Error deleting file {f}: {e}")

def prepare_stage(source_filepath: str) -> str:
    clear_stage()
    filename = os.path.basename(source_filepath)
    dest_path = os.path.join(STAGE_DIR, filename)
    
    _resize_image(source_filepath, dest_path)
    
    print(f"Staged and resized file to {TARGET_SIZE}: {dest_path}")
    return dest_path

def get_random_study_image() -> str:
    search_path = os.path.join(STUDY_DIR, '*', '*.*')
    all_images = glob.glob(search_path)
    valid_images = [
        f for f in all_images 
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
    ]
    if not valid_images:
        raise FileNotFoundError(f"No images found in {STUDY_DIR}")
    return random.choice(valid_images)

def predict_single_image() -> list[tuple[str, float]]:
    staged_files = glob.glob(os.path.join(STAGE_DIR, '*.*'))
    if not staged_files:
        raise FileNotFoundError("No image file found in the 'stage' directory to predict.")
    
    image_to_predict_path = staged_files[0]
    print(f"Predicting on: {image_to_predict_path}")
    predictions = eval.predictSingle(MODEL_PATH, image_to_predict_path)
    time.sleep(1) 
    return predictions


def handle_feedback(feedback_type: str) -> str:
    staged_files = glob.glob(os.path.join(STAGE_DIR, '*.*'))
    if not staged_files:
        print("No file in stage to log feedback for.")
        return "⚠️ Error: No image found in stage to log feedback."

    source_path = staged_files[0]
    
    if feedback_type == "accepted":
        dest_dir_name = "ACCEPTED_FEEDBACK" 
        feedback_word = "Accepted"
    elif feedback_type == "rejected":
        dest_dir_name = "REJECTED_FEEDBACK"
        feedback_word = "Rejected"
    else:
        print(f"Unknown feedback type: {feedback_type}")
        clear_stage()
        return "⚠️ Error: Unknown feedback type."
        
    filename = os.path.basename(source_path)
    
    # --- SIMULATION ONLY ---
    print(f"FEEDBACK SIMULATED: Image '{filename}' would have been moved to '{dest_dir_name}'.")
    
    confirmation_message = (
        f"✅ **Feedback Logged:** The prediction was marked as **{feedback_word}**."
    )
    
    clear_stage() 
    
    return confirmation_message


_init_dirs()