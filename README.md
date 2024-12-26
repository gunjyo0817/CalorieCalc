
# CalorieCalc

*Created by ML Final Project Group44*

This repository contains the code and dataset for a food detection and calorie prediction system. The project workflow involves Object Detection-YOLO, Segmentation-Unet, and Calorie Prediction-regression based on healthy lunch box images.

## Project Structure

The project consists of the following directories:

1. **dataset**: Contains the dataset of healthy lunch box images.
2. **preprocess**: Contains code used for labeling data during the training phase.
3. **Object-Detection_YOLO**: Contains code for the object detection task using YOLO.
4. **Segmentation-Unet**: Contains code for segmenting food items by Unet.
5. **Calorie Prediction**: Contains code for predicting the calorie.

## Testing Workflow

The system follows a multi-step process to predict the calorie content of food items:

**Note**: Each module can be tested independently. To ensure smooth testing, it is recommended to test each one separately before running the entire workflow.

1. **Object Detection (YOLO)**:
   - The input images (`.jpg`/`.png`) are processed by the YOLO model.
   - The model detects food items in the images, generating `.txt` files with the food class and bounding box coordinates.
   
2. **Segmentation (Unet)**:
   - The `.txt` output from YOLO is passed to the Segmentation module.
   - The module segments the detected food items, generating `.xml` files for each segmented food item.

3. **Calorie Prediction (Regression)**:
   - The segmented output `.xml` are then passed to the Calorie Prediction module.
   - The model predicts the calorie content of each food item and outputs the calorie value.

Each folder has its own README. For testing the YOLO-based object detection, follow the instructions in the `Object-Detection_YOLO` folder's `README`.

---

## Extract Training Dataset

To extract the training dataset, please download from the following link:
https://drive.google.com/file/d/1K7idMnmbKNQXPnSuzNRKx9PVzI6U2opf/view?usp=drive_link
