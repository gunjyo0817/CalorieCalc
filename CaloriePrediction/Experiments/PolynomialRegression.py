import os
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import math


ROOT_PATH = "/Users/yichen/Desktop/School/ML/FinalProject/CaloriePrediction/"

train_v2_xml_path = os.path.join(ROOT_PATH, 'final_v2/train_high.xml')
test_v2_xml_path = os.path.join(ROOT_PATH, 'final_v2/v2_test_ground_truth_with_high.xml')
train_image_folder = os.path.join(ROOT_PATH, 'with_cal_origin')
test_v2_image_folder = os.path.join(ROOT_PATH, 'final_v2/v2_test_images')

food_calorie_path = os.path.join(ROOT_PATH, 'food_calorie.xlsx')
filename_calorie_path = os.path.join(ROOT_PATH, 'final_v2/v2_calories.xlsx')
food_calorie_df = pd.read_excel(food_calorie_path)
filename_calorie_df_train = pd.read_excel(filename_calorie_path, sheet_name='train')
filename_calorie_df_test = pd.read_excel(filename_calorie_path, sheet_name='test')

food_calorie_map = {row['food name']: row['kcal/ml'] for _, row in food_calorie_df.iterrows()}
food_class = ["cucumber", "chicken", "shrimp", "brocolli", "bean", "salmon", "pork", "rice", "pumpkin", "egg_dish", "nut", "carrot", "king_oyster", "vegetable", "winter_melon", "mango", "avocado", "baby_corn", "beef", "sprout", "eggplant", "mushroom", "potato", "brocoli", "corn", "toast", "onion", "cabbage", "egg", "green_pepper", "green_bean", "tofu", "fish", "tomato", "lemon", "yam", "high"]

print(food_calorie_map)

# Parse XML file and calculate features
def parse_xml_and_calculate_features(xml_path, image_name):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    features = {cls: 0 for cls in food_class}

    for image in root.findall('image'):
        if image.attrib['name'] == image_name:
            for polygon in image.findall('polygon'):
                label = polygon.attrib['label']
                points = polygon.attrib['points']

                # Ignore "rice" and "purple rice" classes
                if label not in food_class or label == 'rice' or label == 'purple rice':
                    continue

                # Compute area (in pixels²)
                points = np.array([[float(coord) for coord in point.split(',')] for point in points.split(';')])
                area = cv2.contourArea(points.astype(np.int32))

                # Add "high" class to the total calorie
                if label == 'high':
                    features[label] += area
                else:
                    features[label] += area

    feature_vector = [features[cls] for cls in food_class]
    return feature_vector

# Extract training data
def prepare_data(filename_df, folder_path, xml_path):
    X, y = [], []
    for _, row in filename_df.iterrows():
        file_name = str(row['file name'])
        total_calorie = row['total calorie']

        file_name = file_name.split('.')[0]
        image_path = None
        for ext in ['png', 'jpg', 'jpeg']:
            possible_path = os.path.join(folder_path, f"{file_name}.{ext}")
            if os.path.exists(possible_path):
                image_path = possible_path
                break

        if image_path is None:
            print(f"Image not found for {file_name}")
            continue
        if not os.path.exists(xml_path):
            print(f"XML for {file_name} not found!")
            continue

        # Calculate features
        calculated_calories = parse_xml_and_calculate_features(xml_path, os.path.basename(image_path))
        X.append(calculated_calories)
        y.append(total_calorie)

    return np.array(X), np.array(y)

# Extract training and testing data
X_train_full, y_train_full = prepare_data(filename_calorie_df_train, train_image_folder, train_v2_xml_path)
X_test, y_test = prepare_data(filename_calorie_df_test, test_v2_image_folder, test_v2_xml_path)

# Split training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

# Standardize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model = LinearRegression()

model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Sort the predictions
sorted_data = sorted(zip(y_test, predictions.flatten(), filename_calorie_df_test['file name']), key=lambda x: x[0])

# Print sorted predictions
print("\nSorted Predictions:")
for true, pred, filename in sorted_data:
    print(f"Filename: {filename}, Ground Truth: {true}, Prediction: {pred}")

# Calculate metrics
# MAE, RMSE
mae_test = mean_absolute_error(y_test, predictions)
rmse_test = math.sqrt(mean_squared_error(y_test, predictions))
# MAPE
mape_test = np.mean(np.abs((y_test - predictions.flatten()) / y_test)) * 100
# R-Squared
r2_test = 1 - (np.sum((y_test - predictions.flatten())**2) / np.sum((y_test - np.mean(y_test))**2))

print(f"MAE: {mae_test:.2f}, RMSE: {rmse_test:.2f}, MAPE: {mape_test:.2f}%, R^2: {r2_test:.2f}")

# Calculate accuracy
errors = np.abs(y_test - predictions.flatten())
accuracy_50 = np.mean(errors < 50) * 100
accuracy_100 = np.mean(errors < 100) * 100
accuracy_150 = np.mean(errors < 150) * 100

print(f"Accuracy (<50): {accuracy_50:.2f}%")
print(f"Accuracy (<100): {accuracy_100:.2f}%")
print(f"Accuracy (<150): {accuracy_150:.2f}%")


def plot_trend_chart(y_test, predictions):
    indices = range(len(y_test))

    plt.figure(figsize=(10, 6))
    plt.plot(indices, y_test, label='True Value', color='blue', linestyle='--', marker='o')  # 真實值
    plt.plot(indices, predictions.flatten(), label='Predicted Value', color='red', linestyle='--', marker='o')  # 預測值
    plt.title('True Values vs Predicted Values', fontsize=16)
    plt.xlabel('Sample Index', fontsize=14)
    plt.ylabel('Calories', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)

plot_trend_chart(y_test, predictions)

output_text_path = f"PolynomialRegression_{int(rmse_test)}.txt"
with open(output_text_path, "w") as f:
    f.write(f"MAE: {mae_test:.2f}, RMSE: {rmse_test:.2f}, MAPE: {mape_test:.2f}%, R^2: {r2_test:.2f}\n")
    f.write(f"Accuracy (<50): {accuracy_50:.2f}%\n")
    f.write(f"Accuracy (<100): {accuracy_100:.2f}%\n")
    f.write(f"Accuracy (<150): {accuracy_150:.2f}%\n")
    f.write("\nSorted Predictions:\n")
    for true, pred, filename in sorted_data:
        f.write(f"Filename: {filename}, Ground Truth: {true}, Prediction: {pred}\n")

output_path = f"PolynomialRegression_{int(rmse_test)}.png"
plt.savefig(output_path, dpi=300)
print(f"Regression plot saved as {output_path}")