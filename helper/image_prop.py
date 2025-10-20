import os
import shutil
from sklearn.model_selection import train_test_split
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

RANDOM_SEED=42

def split():
    source_dir='baseDataset'
    target_dir='splitDataset'

    splits = ['train', 'val', 'study']

    for split in splits:
        for class_name in os.listdir(source_dir):
            class_path = os.path.join(source_dir, class_name)
            if os.path.isdir(class_path):
                os.makedirs(os.path.join(target_dir, split, class_name), exist_ok=True)

    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        images = [os.path.join(class_path, f) for f in os.listdir(class_path)]

        train_files, temp_files = train_test_split(images, test_size=0.3, random_state=RANDOM_SEED)
        val_files, study_files = train_test_split(temp_files, test_size=1/3, random_state=RANDOM_SEED)

        split_mp = {'train': train_files, 'val': val_files, 'study': study_files}

        for split, files in split_mp.items():
            for file in files:
                file_name = os.path.basename(file)
                shutil.copy(file, os.path.join(target_dir, split, class_name, file_name))

def visualiseImageData():
    images = []
    train_dir = 'splitDataset/train' 
    val_dir = 'splitDataset/val'
    for class_name in os.listdir(train_dir):
        class_dir = os.path.join(train_dir, class_name)
        for filename in os.listdir(class_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                images.append(os.path.join(class_dir, filename))

    for class_name in os.listdir(val_dir):
        class_dir = os.path.join(val_dir, class_name)
        for filename in os.listdir(class_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                images.append(os.path.join(class_dir, filename))

    w, h = [], []

    for image in images:
        with Image.open(image) as img:
            x,y = img.size
            w.append(x)
            h.append(y)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(w, bins=30, color='skyblue', edgecolor='black')
    plt.title('Image Width Distribution')
    plt.xlabel('Width (pixels)')
    plt.ylabel('Number of Images')

    plt.subplot(1, 2, 2)
    plt.hist(h, bins=30, color='salmon', edgecolor='black')
    plt.title('Image Height Distribution')
    plt.xlabel('Height (pixels)')
    plt.ylabel('Number of Images')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
# split()
    visualiseImageData()