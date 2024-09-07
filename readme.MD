# Shooting Sound Recognition

## Overview
This project aims to develop a machine learning model to recognize shooting sounds from audio files. The model is built using a Convolutional Neural Network (CNN) and trained on Mel-frequency cepstral coefficients (MFCCs) extracted from audio recordings of gunshots and ambient sounds.

## Project Structure
- **ss_recognition.py**: Main script for loading data, extracting features, training, and evaluating the model.
- **ss_recognition - Copy.py**: Backup copy of the main script.
- **ss_recognition.ipynb**: Jupyter notebook for interactive development and testing.
- **.gitignore**: Specifies files and directories to be ignored by Git.

## Data
The dataset consists of audio files categorized into gunshots and ambient sounds:
- **Gunshots (1)/Gunshots/Automatic**: Contains automatic gunshot sounds.
- **Gunshots (1)/Gunshots/Single Shot**: Contains single shot gun sounds.
- **1_Ambience Sound/Ambience Sound**: Contains ambient sounds.

## Key Functions
### Feature Extraction
Extracts MFCC features from audio files.