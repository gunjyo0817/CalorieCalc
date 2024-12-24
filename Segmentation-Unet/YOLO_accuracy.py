import xml.etree.ElementTree as ET
import numpy as np
import cv2
from collections import defaultdict

def parse_xml_to_polygons(xml_path):
    """
    Parse an XML file to extract polygons grouped by file and label.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    file_polygons = {}
    file_sizes = {}
    
    for image in root.findall('image'):
        file_name = image.get('name')
        img_width = int(image.get('width'))
        img_height = int(image.get('height'))
        
        polygons_by_label = defaultdict(list)
        
        for polygon in image.findall('polygon'):
            label = polygon.get('label')
            points = polygon.get('points')
            
            # Convert points to a list of (x, y) tuples
            points = np.array([list(map(lambda x: int(float(x)), point.split(','))) for point in points.split(';')], dtype=np.int32)
            polygons_by_label[label].append(points)
        
        file_polygons[file_name] = polygons_by_label
        file_sizes[file_name] = (img_width, img_height)
    
    return file_polygons, file_sizes


def calculate_iou(mask1, mask2):
    """
    Calculate IoU between two binary masks.
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0

def generate_mask_from_polygons(polygons, img_width, img_height):
    """
    Generate a binary mask from a list of polygons.
    """
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    for polygon in polygons:
        cv2.fillPoly(mask, [polygon], 1)
    return mask

def calculate_iou_for_files(gt_polygons, pred_polygons, img_width, img_height):
    """
    Calculate IoU for all labels in a file.
    """
    ious = {}
    all_labels = set(gt_polygons.keys()).union(pred_polygons.keys())
    
    for label in all_labels:
        # Get polygons for the label (default to empty list)
        gt_label_polygons = gt_polygons.get(label, [])
        pred_label_polygons = pred_polygons.get(label, [])
        
        # Generate masks for the label
        gt_mask = generate_mask_from_polygons(gt_label_polygons, img_width, img_height)
        pred_mask = generate_mask_from_polygons(pred_label_polygons, img_width, img_height)
        
        # Calculate IoU
        iou = calculate_iou(gt_mask, pred_mask)
        ious[label] = iou
    
    return ious

def calculate_iou_for_xmls(gt_xml, pred_xml):
    """
    Calculate IoU for each file in the XMLs, considering dynamic image sizes.
    """
    gt_data, gt_sizes = parse_xml_to_polygons(gt_xml)
    pred_data, pred_sizes = parse_xml_to_polygons(pred_xml)
    
    all_files = set(gt_data.keys()).union(pred_data.keys())
    file_ious = {}
    
    for file_name in all_files:
        gt_polygons = gt_data.get(file_name, {})
        pred_polygons = pred_data.get(file_name, {})
        
        # Use the size from ground truth or prediction, fallback to (1024, 768) if unknown
        img_width, img_height = gt_sizes.get(file_name, pred_sizes.get(file_name, (1024, 768)))
        
        ious = calculate_iou_for_files(gt_polygons, pred_polygons, img_width, img_height)
        file_ious[file_name] = ious
    
    return file_ious

def calculate_mean_iou_for_files(gt_polygons, pred_polygons, img_width, img_height):
    """
    Calculate mean IoU for a file, as well as IoUs for each label.
    """
    ious = {}
    all_labels = set(gt_polygons.keys()).union(pred_polygons.keys())
    label_ious = []

    for label in all_labels:
        # Get polygons for the label (default to empty list)
        gt_label_polygons = gt_polygons.get(label, [])
        pred_label_polygons = pred_polygons.get(label, [])

        # Generate masks for the label
        gt_mask = generate_mask_from_polygons(gt_label_polygons, img_width, img_height)
        pred_mask = generate_mask_from_polygons(pred_label_polygons, img_width, img_height)

        # Calculate IoU
        iou = calculate_iou(gt_mask, pred_mask)
        ious[label] = iou
        label_ious.append(iou)

    # Mean IoU for the file
    mean_iou = np.mean(label_ious) if label_ious else 0
    return mean_iou, ious

def calculate_pixel_accuracy(gt_mask, pred_mask):
    """
    Calculate pixel accuracy between two binary masks.
    """
    correct_pixels = np.sum(gt_mask == pred_mask)
    total_pixels = gt_mask.size
    return correct_pixels / total_pixels if total_pixels > 0 else 0

def calculate_pixel_accuracy_for_files(gt_polygons, pred_polygons, img_width, img_height):
    """
    Calculate pixel accuracy for all polygons in a file.
    """
    # Generate binary masks for ground truth and prediction
    gt_mask = generate_mask_from_polygons([polygon for polygons in gt_polygons.values() for polygon in polygons], img_width, img_height)
    pred_mask = generate_mask_from_polygons([polygon for polygons in pred_polygons.values() for polygon in polygons], img_width, img_height)
    
    # Calculate pixel accuracy
    pixel_accuracy = calculate_pixel_accuracy(gt_mask, pred_mask)
    return pixel_accuracy


def calculate_performance_for_xmls(gt_xml, pred_xml):
    """
    Calculate IoU for each file in the XMLs and compute the mean IoU across all files.
    """
    gt_data, gt_sizes = parse_xml_to_polygons(gt_xml)
    pred_data, pred_sizes = parse_xml_to_polygons(pred_xml)

    all_files = set(gt_data.keys()).union(pred_data.keys())
    file_ious = {}
    mean_ious = []
    file_pixel_accuracies = {}
    all_pixel_accuracies = []


    for file_name in all_files:
        gt_polygons = gt_data.get(file_name, {})
        pred_polygons = pred_data.get(file_name, {})

        # Use the size from ground truth or prediction, fallback to (1024, 768) if unknown
        img_width, img_height = gt_sizes.get(file_name, pred_sizes.get(file_name, (1024, 768)))

        # Calculate mean IoU and IoUs for each label in the file
        mean_iou, ious = calculate_mean_iou_for_files(gt_polygons, pred_polygons, img_width, img_height)
        file_ious[file_name] = {"mean_iou": mean_iou, "ious": ious}
        mean_ious.append(mean_iou)

        pixel_accuracy = calculate_pixel_accuracy_for_files(gt_polygons, pred_polygons, img_width, img_height)
        file_pixel_accuracies[file_name] = pixel_accuracy
        all_pixel_accuracies.append(pixel_accuracy)

    # Calculate overall mean IoU
    overall_mean_iou = np.mean(mean_ious) if mean_ious else 0
    overall_pixel_accuracy = np.mean(all_pixel_accuracies) if all_pixel_accuracies else 0

    return overall_mean_iou, file_ious, overall_pixel_accuracy, file_pixel_accuracies


# Example Usage
if __name__ == "__main__":
    gt_xml_path = "../data/xml_groundtruth/v2_test_with_high.xml"
    pred_xml_path = "./output/xml_files/v2_test_resnet101.xml"

    overall_mean_iou, file_ious, overall_pixel_accuracy, file_pixel_accuracies = calculate_performance_for_xmls(gt_xml_path, pred_xml_path)

    # Print per-file IoUs
    for file_name, iou_data in file_ious.items():
        print(f"File: {file_name}")
        print(f"  Mean IoU: {iou_data['mean_iou']:.4f}")
        print(f"  Pixel Accuracy: {iou_data['mean_iou']:.4f}")
        for label, iou in iou_data["ious"].items():
            print(f"    Label: {label}, IoU: {iou:.4f}")

    # Print overall mean IoU
    print("===========================================")
    print(f"Overall Mean IoU: {overall_mean_iou:.4f}")

    
    print(f"\nOverall pixel accuracy: {overall_pixel_accuracy:.4f}")
    print("===========================================")
    for file_name, accuracy in file_pixel_accuracies.items():
        print(f"File: {file_name}, Pixel Accuracy: {accuracy:.4f}")