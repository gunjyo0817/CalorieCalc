import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import xml.etree.ElementTree as ET
from xml.dom import minidom

IMG_WIDTH, IMG_HEIGHT = 256, 256  # Model input size

class_dict = {
    0: 'cucumber', 1: 'chicken', 2: 'shrimp', 3: 'brocolli', 4: 'bean',
    5: 'salmon', 6: 'pork', 7: 'rice', 8: 'pumpkin', 9: 'egg_dish',
    10: 'nut', 11: 'carrot', 12: 'king_oyster', 13: 'vegetable',
    14: 'winter_melon', 15: 'mango', 16: 'avocado', 17: 'baby_corn',
    18: 'beef', 19: 'sprout', 20: 'eggplant', 21: 'mushroom',
    22: 'potato', 23: 'brocoli', 24: 'corn', 25: 'toast',
    26: 'onion', 27: 'cabbage', 28: 'egg', 29: 'green_pepper',
    30: 'green_bean', 31: 'tofu', 32: 'fish', 33: 'tomato',
    34: 'lemon', 35: 'yam', 36: 'high'
}


def yolo_to_pixel_coords(yolo_box, img_width, img_height):
    """
    Convert YOLO relative coordinates to pixel coordinates.
    """
    class_id, x_center, y_center, width, height = yolo_box
    x_center, y_center, width, height = (
        float(x_center) * img_width,
        float(y_center) * img_height,
        float(width) * img_width,
        float(height) * img_height,
    )
    x_min = int(x_center - width / 2)
    y_min = int(y_center - height / 2)
    x_max = int(x_center + width / 2)
    y_max = int(y_center + height / 2)
    return class_id, x_min, y_min, x_max, y_max


def load_yolo_boxes(yolo_txt_path, img_width, img_height):
    """
    Load YOLO boxes from a text file.
    """
    boxes = []
    with open(yolo_txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            boxes.append(yolo_to_pixel_coords(parts, img_width, img_height))
    return boxes


def predict_and_restore_mask(image, model, box, img_width, img_height):
    """
    Predict and restore mask for a single bounding box. If the mask is empty (all 0),
    the entire bounding box is filled with 1.
    """
    class_id, x_min, y_min, x_max, y_max = box
    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(img_width, x_max), min(img_height, y_max)

    # Crop and preprocess the image
    cropped_image = image[y_min:y_max, x_min:x_max]
    resized_image = cv2.resize(cropped_image, (IMG_WIDTH, IMG_HEIGHT)) / 255.0  # Normalize
    input_image = np.expand_dims(resized_image, axis=0)  # Add batch dimension

    # Predict the mask
    predicted_mask = model.predict(input_image)[0, :, :, 0]  # Output shape: (256, 256)
    restored_mask = cv2.resize(predicted_mask, (x_max - x_min, y_max - y_min), interpolation=cv2.INTER_NEAREST)
    restored_mask = (restored_mask > 0.5).astype(np.uint8)  # Threshold to binary mask

    # Check if the mask is empty (all zeros)
    if np.sum(restored_mask) == 0:
        print(f"Empty mask detected for bounding box: ({x_min}, {y_min}, {x_max}, {y_max}). Filling with 1.")
        restored_mask[:] = 1  # Fill the entire bounding box with 1

    # Create the full-size mask and insert the restored mask into its location
    full_mask = np.zeros((img_height, img_width), dtype=np.uint8)
    full_mask[y_min:y_max, x_min:x_max] = restored_mask

    return full_mask, class_id

# def mask_to_polygons(mask, class_id):
#     """
#     Convert a binary mask to a single polygon by merging all contours for the same class.
#     """
#     polygons = []
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # 合併所有輪廓
#     merged_contour = np.vstack([contour for contour in contours if cv2.contourArea(contour) > 10])

#     # 將合併後的輪廓轉換為多邊形格式
#     points = ";".join([f"{int(x)},{int(y)}" for x, y in merged_contour.reshape(-1, 2)])
#     polygons.append({"class": class_dict[int(class_id)], "points": points})
    
#     return polygons



def mask_to_polygons(mask, class_id):
    """
    Convert a binary mask to a list of polygons.
    """
    polygons = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 10:  # Filter small areas
            points = ";".join([f"{int(x)},{int(y)}" for x, y in contour.reshape(-1, 2)])
            polygons.append({"class": class_dict[int(class_id)], "points": points})
    return polygons


def save_mask_as_png(mask, output_path):
    """
    Save the final combined mask as a PNG file.
    """
    cv2.imwrite(output_path, mask * 255)


def generate_combined_xml_with_polygons(image_infos, xml_output_path):
    """
    Generate a single XML file containing information for all images with polygons for each object.
    """
    root = ET.Element("annotations")
    for info in image_infos:
        image_elem = ET.SubElement(root, "image", {
            "id": str(info["id"]),
            "name": info["filename"],
            "width": str(info["width"]),
            "height": str(info["height"]),
        })
        for polygon_info in info["polygons"]:
            ET.SubElement(image_elem, "polygon", {
                "label": str(polygon_info["class"]),
                "points": polygon_info["points"],
            })
    xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="   ")
    with open(xml_output_path, "w") as f:
        f.write(xml_str)


def process_folder_to_single_xml(image_folder, yolo_folder, model_path, output_mask_folder, output_xml_path):
    """
    Process an entire folder of images, predict masks, and save outputs to a single XML file.
    """
    os.makedirs(output_mask_folder, exist_ok=True)
    model = load_model(model_path)
    image_infos = []
    image_id = 0

    for image_file in sorted(os.listdir(image_folder)):
        if not image_file.lower().endswith(('.jpg', '.png', 'jpeg')):
            continue

        image_path = os.path.join(image_folder, image_file)
        yolo_txt_path = os.path.join(yolo_folder, os.path.splitext(image_file)[0] + ".txt")
        output_mask_path = os.path.join(output_mask_folder, os.path.splitext(image_file)[0] + "_mask.png")

        if not os.path.exists(yolo_txt_path):
            print(f"YOLO file not found for {image_file}, skipping...")
            continue

        image = cv2.imread(image_path)
        img_height, img_width, _ = image.shape
        yolo_boxes = load_yolo_boxes(yolo_txt_path, img_width, img_height)
        final_mask = np.zeros((img_height, img_width), dtype=np.uint8)
        polygons = []

        for box in yolo_boxes:
            box_mask, class_id = predict_and_restore_mask(image, model, box, img_width, img_height)
            final_mask = np.maximum(final_mask, box_mask)
            polygons.extend(mask_to_polygons(box_mask, class_id))

        save_mask_as_png(final_mask, output_mask_path)
        image_infos.append({
            "id": image_id,
            "filename": image_file,
            "width": img_width,
            "height": img_height,
            "polygons": polygons,
        })
        print(f"Processed {image_file}: Mask saved to {output_mask_path}")
        image_id += 1

    generate_combined_xml_with_polygons(image_infos, output_xml_path)
    print(f"Combined XML saved to {output_xml_path}")


if __name__ == "__main__":
    process_folder_to_single_xml(
        image_folder="../data/test/v2_test_images",
        yolo_folder="../data/test/yolo_boxes/v2_test_txt_after_yolo",
        model_path="./weight/unet_yolo_earlystop_restet101_342image.h5",
        output_mask_folder="./output/v2_test",
        output_xml_path="./output/xml_files/v2_test_resnet101.xml"
    )
