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

### Load Audio Files
Loads audio files from a directory and extracts features.

## Data Preparation
Combines and shuffles the data, then splits it into training and testing sets.

## Model
A Convolutional Neural Network (CNN) is built and trained to classify the audio features.

## Evaluation
The model is evaluated on the test set, and performance metrics are printed.


## Usage
1. Ensure you have the required libraries installed:
    ```sh
    pip install numpy pandas librosa tensorflow scikit-learn
    ```
2. Run the main script:
    ```sh
    python ss_recognition.py
    ```

## Results
The model achieves high accuracy in distinguishing between gunshot and ambient sounds, as indicated by the evaluation metrics.

## License
This project is licensed under the MIT License.