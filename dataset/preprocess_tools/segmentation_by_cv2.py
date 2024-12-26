import cv2
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import rcParams
from matplotlib.cm import get_cmap
class_id_to_label = {
    0: 'cucumber', 
    1: 'chicken',
    2: 'shrimp',
    3: 'brocolli',
    4: 'bean',
    5: 'salmon',
    6: 'pork',
    7: 'rice',
    8: 'pumpkin',
    9: 'egg_dish',
    10: 'nut',
    11: 'carrot',
    12: 'king_oyster',
    13: 'vegetable',
    14: 'winter_melon',
    15: 'mango',
    16: 'avocado',
    17: 'baby_corn',
    18: 'beef',
    19: 'sprout',
    20: 'eggplant',
    21: 'mushroom',
    22: 'potato',
    23: 'brocoli',
    24: 'corn',
    25: 'toast',
    26: 'onion',
    27: 'cabbage',
    28: 'egg',
    29: 'green_pepper',
    30: 'green_bean',
    31: 'tofu',
    32: 'fish',
    33: 'tomato',
    34: 'lemon',
    35: 'yam'
}
# Set the font to one that supports Chinese characters
# rcParams['font.sans-serif'] = ['SimHei']  # You can also use 'Microsoft YaHei' or other Chinese-supporting fonts
# rcParams['axes.unicode_minus'] = False    # To ensure minus signs are displayed correctly

def parse_yolo_annotation(txt_path, image_width, image_height):
    """Parse YOLO annotation file and convert it to bounding box format."""
    objects = []
    with open(txt_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])  # Class ID (integer)
            center_x = float(parts[1]) * image_width  # x center, scaled by image width
            center_y = float(parts[2]) * image_height  # y center, scaled by image height
            width = float(parts[3]) * image_width  # width, scaled by image width
            height = float(parts[4]) * image_height  # height, scaled by image height
            
            xmin = int(center_x - width / 2)
            ymin = int(center_y - height / 2)
            xmax = int(center_x + width / 2)
            ymax = int(center_y + height / 2)
            
            objects.append({"name": str(class_id), "bbox": (xmin, ymin, xmax, ymax)})
    return objects

def auto_select_color_range_kmeans(image, k=3):
    """使用 K-means 聚类自动选择颜色范围"""
    # 转换到 RGB 空间并将其重塑为一个二维数组
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = rgb_image.reshape((-1, 3))

    # 使用 K-means 聚类分析颜色
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(np.float32(pixels), k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # 选择聚类中心作为前景颜色
    dominant_color = centers[np.argmax(np.bincount(labels.flatten()))]
    
    # 根据聚类结果计算颜色范围
    lower_bound = np.array([max(0, int(dominant_color[0]) - 30), 50, 50])
    upper_bound = np.array([min(255, int(dominant_color[0]) + 30), 255, 255])

    return lower_bound, upper_bound

def refine_initial_mask(image, rect):
    """根据颜色空间和边缘检测自动初始化更精确的 GrabCut Mask"""
    mask = np.zeros(image.shape[:2], np.uint8)

    # 自动选择颜色范围
    lower_bound, upper_bound = auto_select_color_range_kmeans(image)

    # 转换到 HSV 色彩空间以区分目标和背景
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 使用自动选择的颜色范围
    mask_color = cv2.inRange(hsv_image, lower_bound, upper_bound)
    mask[mask_color > 0] = cv2.GC_PR_FGD

    # 使用边缘检测改进边界区域
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
    edges = cv2.Canny(gray_image, 100, 200)
    # edges = cv2.Canny(gray_image, 50, 150)
    mask[edges > 0] = cv2.GC_BGD

    # 合并用户定义的矩形区域
    xmin, ymin, w, h = rect
    mask[ymin:ymin + h, xmin:xmin + w] = cv2.GC_PR_FGD

    # 可选：扩展前景区域
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    return mask

def smooth_segmentation(mask):
    """对分割的 Mask 进行后处理"""
    kernel = np.ones((5, 5), np.uint8)

    # 先膨胀后腐蚀去除小块噪声
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 可选：模糊处理
    mask = cv2.GaussianBlur(mask.astype("float32"), (5, 5), 0)
    mask = (mask > 0.5).astype("uint8")

    return mask

def apply_grabcut(image, rect, iterations=10):
    """动态初始化和改进 GrabCut 的背景移除功能"""
    mask = refine_initial_mask(image, rect)
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)

    # 进行 GrabCut 迭代
    cv2.grabCut(image, mask, None, bg_model, fg_model, iterations, cv2.GC_INIT_WITH_MASK)
    
    # 提取最终结果，将背景区域（0, 2）设为背景，将前景区域（1, 3）设为前景
    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")

    # 后处理优化分割结果
    mask = smooth_segmentation(mask)

    return mask


# def apply_grabcut(image, rect, iterations=10):
#     """Apply GrabCut algorithm to segment object."""
#     mask = np.zeros(image.shape[:2], np.uint8)
#     bg_model = np.zeros((1, 65), np.float64)
#     fg_model = np.zeros((1, 65), np.float64)
#     cv2.grabCut(image, mask, rect, bg_model, fg_model, iterations, cv2.GC_INIT_WITH_RECT)
#     mask = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
#     return mask


def save_annotation_to_xml(output_path, objects, output_image, image_name):
    """Save updated annotation to a new XML file."""
    annotation = ET.Element("annotation")
    filename = ET.SubElement(annotation, "filename")
    filename.text = image_name
    size = ET.SubElement(annotation, "size")
    width = ET.SubElement(size, "width")
    width.text = str(output_image.shape[1])
    height = ET.SubElement(size, "height")
    height.text = str(output_image.shape[0])
    
    for obj in objects:
        obj_elem = ET.SubElement(annotation, "object")
        name = ET.SubElement(obj_elem, "name")
        name.text = obj["name"]
        
        poly = ET.SubElement(obj_elem, "poly")
        for point in obj["poly"]:
            pt_elem = ET.SubElement(poly, "point")
            pt_elem.text = f"{point[0]},{point[1]}"
    
    rough_string = ET.tostring(annotation, "utf-8")
    reparsed = minidom.parseString(rough_string)
    with open(output_path, "w") as f:
        f.write(reparsed.toprettyxml(indent="  "))

def remove_background_with_annotation(image_path, txt_path, image_width, image_height):
    """Perform background removal and generate polygon annotations."""
    image = cv2.imread(image_path)
    original_height, original_width = image.shape[:2]
    output = np.zeros((original_height, original_width, 4), dtype=np.uint8)
    objects = parse_yolo_annotation(txt_path, image_width, image_height)

    for obj in objects:
        bbox = obj['bbox']
        xmin, ymin, xmax, ymax = bbox

        if xmax - xmin <= 1 or ymax - ymin <= 1:
            print(f"Skipping invalid region {xmin}, {ymin}, {xmax}, {ymax}")
            continue

        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(original_width - 1, xmax)
        ymax = min(original_height - 1, ymax)
        
        rect = (xmin, ymin, xmax - xmin, ymax - ymin)
        mask = apply_grabcut(image, rect)
        result = image * mask[:, :, np.newaxis]
        output[:, :, :3] = np.where(mask[:, :, np.newaxis], result, output[:, :, :3])
        output[:, :, 3] = np.where(mask, 255, output[:, :, 3])

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            poly_points = contour.reshape(-1, 2).tolist()
            obj['poly'] = poly_points
        else:
            obj['poly'] = []

    output_annotation_path = 'output_annotation.xml'
    save_annotation_to_xml(output_annotation_path, objects, output, image_name=image_path)
    return output, output_annotation_path

def convert_xml(input_file, output_file):
    """Convert XML format to a different structure with proper formatting and ignore empty points."""
    try:
        tree = ET.parse(input_file)
        root = tree.getroot()
        new_root = ET.Element("annotations")
        filename = root.find('filename')
        size = root.find('size')

        if filename is not None and size is not None:
            width = size.find('width')
            height = size.find('height')

            new_image = ET.SubElement(new_root, "image")
            new_image.set("name", os.path.basename(filename.text))
            new_image.set("width", width.text)
            new_image.set("height", height.text)

            for obj in root.findall('object'):
                label = obj.find('name').text
                poly = obj.find('poly')
                points = ";".join(
                    point.text.strip()
                    for point in poly.findall('point')
                    if point.text.strip()
                )
                if points:  # Only add polygon if points are not empty
                    polygon = ET.SubElement(new_image, "polygon", {
                        "label": label,
                        "source": "semi-auto",
                        "occluded": "0",
                        "points": points,
                        "z_order": "0"
                    })

        # Create a well-formatted string with pretty print
        rough_string = ET.tostring(new_root, encoding="unicode")
        reparsed = minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent="  ")

        # Save the formatted XML to the output file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(pretty_xml)
        print(f"Converted XML saved to {output_file}")
    except Exception as e:
        print(f"Error: {e}")

import os
from PIL import Image

def process_all_annotations(input_folder):
    """Process all annotations in a folder."""
    # 支持的图像扩展名
    valid_image_extensions = ['.jpg', '.jpeg', '.png']

    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):  # 只处理 .txt 文件
            txt_path = os.path.join(input_folder, filename)
            image_base_path = os.path.splitext(txt_path)[0]

            # 遍历支持的图像扩展名，检查是否存在对应的图像文件
            image_folder = input_folder
            image_path = None
            for ext in valid_image_extensions:
                candidate_image_path = os.path.join(image_folder, os.path.basename(image_base_path) + ext)
                if os.path.exists(candidate_image_path):
                    image_path = candidate_image_path
                    break

            if not image_path:
                print(f"No image file found for {txt_path}")
                continue

            # 获取图像大小
            image_width, image_height = Image.open(image_path).size

            output_image_path = os.path.join(input_folder, f"{filename[:-4]}_annotated_image.png")
            output_xml_path = os.path.join(input_folder, f"converted_{filename[:-4]}.xml")

            if os.path.exists(output_image_path) and os.path.exists(output_xml_path):
                print(f"Skipping {filename}, outputs already exist.")
                continue

            print(f"Processing: {filename}")
            output_image, output_annotation_path = remove_background_with_annotation(image_path, txt_path, image_width, image_height)
            convert_xml(output_annotation_path, output_xml_path)
            visualize_annotation(output_xml_path, input_folder, filename)

def visualize_annotation(xml_file, input_folder, file_name):
    # 載入 XML 文件
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # 獲取圖像信息
    image_element = root.find('image')
    image_path = image_element.get('name')  # 圖像文件路徑
    image_width = int(image_element.get('width'))
    image_height = int(image_element.get('height'))

    # 加載圖像
    image = Image.open(input_folder + '/' + image_path)

    # 創建繪圖窗口
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)

    # 解析多邊形標註
    labels = set(polygon.get('label') for polygon in image_element.findall('polygon'))  # 獲取所有標籤
    color_map = get_cmap('tab10')  # 使用 Matplotlib 的內建顏色地圖
    label_colors = {label: color_map(i / len(labels)) for i, label in enumerate(labels)}  # 分配顏色

    for polygon in image_element.findall('polygon'):
        label = polygon.get('label')
        print(f"Label found in polygon: {label}")  # Debug print
        class_id = [key for key, value in class_id_to_label.items() if value == label]
        print(f"Matching class_id: {class_id}")  # Debug print
        
        class_id = class_id[0] if class_id else -1  # Default to -1 if not found
        
        points = polygon.get('points')  # 點的座標
        points = [tuple(map(int, p.split(','))) for p in points.split(';')]

        # 繪製多邊形
        polygon_patch = patches.Polygon(points, closed=True, fill=False, edgecolor=label_colors[label], linewidth=5)
        ax.add_patch(polygon_patch)
        ax.text(points[0][0], points[0][1], class_id_to_label.get(int(label), 'unknown'), color=label_colors[label], fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

    # 添加圖例
    legend_patches = [patches.Patch(color=label_colors[label], label=class_id_to_label.get(int(label), 'unknown')) for label in labels]
    ax.legend(handles=legend_patches, loc='upper right')

    # 顯示
    plt.axis('off')

    # 儲存標註過後的圖片（如果需要）
    output_image_path = 'annotated_image.png'
    plt.savefig(f"{input_folder}/{file_name[:-4]}_{output_image_path}", bbox_inches='tight', pad_inches=0, transparent=True)

    # 關閉繪圖窗口
    plt.close()

# Main function to process all annotations
input_folder = "test_image_and_txt"  # Specify your folder path
process_all_annotations(input_folder)
