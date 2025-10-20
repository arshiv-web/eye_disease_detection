# eye_disease_detection
eye_disease_detection

This project implements a complete Machine Learning pipeline for the multi-class classification of five common eye conditions from ocular images. The system is engineered for reproducibility, reliability, and application readiness.

# Clone Repo: 
git clone https://github.com/arshiv-web/eye_disease_detection.git
cd eye_disease_detection

# Install Dependencies: (Ensure all required libraries are installed)
pip install -r requirements.txt

# Training the model
python train.py

# Evaluating a trained model
1. On the study dataset (splitDataset/study): Call the predictStudyDataset methods with model chkpt in evaluate.py
2. On a new image: Call the predictSingle method with model chkpt in evaluate.py 

# Running the UI
python ui.py


If any questions reach out to arshiv@umich.edu
