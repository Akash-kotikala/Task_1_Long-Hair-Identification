# Task_1_Long-Hair-Identification
ML model to predict gender: long hair → female, short-haired females → male (ages 20–30); standard prediction otherwise. Uses ResNet18, OpenCV, Streamlit GUI. Eval: accuracy, F1, confusion matrix.

### Link for hair_length_model:: 
https://drive.google.com/file/d/1aIPclFVAxenUbcfhylByFb-JNWY2iuzK/view?usp=drive_link

### Link for age_gender model::  
https://drive.google.com/file/d/1rWM2BoT5K99uA4mqmMwfWrQp_VfZDRgG/view?usp=drive_link


### Link for dataset :: 

https://www.kaggle.com/datasets/moritzm00/utkface-cropped



# Hair Length Detector

## Problem Statement
Detect and classify hair length (long vs. short) in facial images. This can be useful for applications like virtual try-ons, demographic analysis, or augmented reality filters. The challenge is to accurately segment hair from faces and classify based on masks, handling variations in lighting, angles, and demographics.

## Dataset
- **Source**: UTKFace dataset (23,705 images of faces with age, gender, and ethnicity labels).
- **Preprocessing**: Images resized to 200x200, hair masks generated via segmentation. Filtered to ~10,000 samples for training.
- **Classes**: Binary (long hair: 1, short/no hair: 0).
- **Download**: [Kaggle UTKFace](https://www.kaggle.com/datasets/jangedoo/utkface-new).
- **Size**: ~1GB uncompressed.

## Methodology
1. **Data Loading & Preprocessing**: Load UTKFace images, generate hair masks using pre-trained segmentation (e.g., via MediaPipe or custom). Normalize images to [0,1].
2. **Model**: Transfer learning with EfficientNetB0 base + custom head for binary classification.
   - Input: 200x200x3 images.
   - Output: Sigmoid for binary class.
   - Optimizer: Adam (lr=0.001).
   - Loss: Binary Crossentropy.
   - Metrics: Accuracy, Precision, Recall.
3. **Training**: 80/20 train-test split, 20 epochs with early stopping. Batch size: 32.
4. **Evaluation**: Confusion matrix, accuracy on test set.
5. **Tools**: TensorFlow/Keras, OpenCV, Pandas, Matplotlib.

## Results
- **Accuracy**: ~92% on test set (based on notebook logs).
- **Precision/Recall**: High for long hair (~0.95), slightly lower for short (~0.88) due to class imbalance.
- **Sample Output**: Model predicts hair length on new images; outputs zipped (model, masks, plots).
- **Limitations**: Struggles with occluded hair or unusual styles; could improve with more diverse data.

## Installation
```bash
pip install tensorflow opencv-python pandas matplotlib scikit-learn
