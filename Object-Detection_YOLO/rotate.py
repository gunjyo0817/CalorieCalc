import xml.etree.ElementTree as ET
import cv2
import numpy as np
import math
import os
import shutil

def rotate_point(x, y, angle, cx, cy):
    """以(cx, cy)為中心旋轉點 (x, y)"""
    rad = math.radians(angle)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    x_new = cos_a * (x - cx) - sin_a * (y - cy) + cx
    y_new = sin_a * (x - cx) + cos_a * (y - cy) + cy
    return x_new, y_new

def rotate_polygon(points, angle, cx, cy):
    """旋轉多邊形"""
    rotated = []
    for x, y in points:
        x_new, y_new = rotate_point(x, y, angle, cx, cy)
        rotated.append((x_new, y_new))
    return rotated

def parse_polygon_points(points_str):
    """解析 <polygon points="x,y x,y ..."> 為座標列表"""
    points = []
    for point in points_str.strip().split(";"):
        x, y = map(float, point.split(","))
        points.append((x, y))
    return points

def format_polygon_points(points):
    """將座標列表格式化為 <polygon points="x,y x,y ...">"""
    return ";".join(f"{int(x)},{int(y)}" for x, y in points)

def rotate_image(image, angle):
    """旋轉圖片"""
    (h, w) = image.shape[:2]
    cx, cy = w // 2, h // 2
    # 計算旋轉矩陣
    rotation_matrix = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    # 計算旋轉後的邊界
    cos_a = abs(rotation_matrix[0, 0])
    sin_a = abs(rotation_matrix[0, 1])
    new_w = int((h * sin_a) + (w * cos_a))
    new_h = int((h * cos_a) + (w * sin_a))
    # 調整旋轉矩陣以考慮新邊界
    rotation_matrix[0, 2] += (new_w / 2) - cx
    rotation_matrix[1, 2] += (new_h / 2) - cy
    # 執行旋轉
    rotated_image = cv2.warpAffine(image, rotation_matrix, (new_w, new_h))
    return rotated_image, rotation_matrix

def load_image_with_chinese_path(file_path):
    """加载包含中文路径的图片"""
    with open(file_path, 'rb') as f:
        image_data = bytearray(f.read())
        image_np = np.asarray(image_data, dtype=np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    return image

def save_image_with_chinese_name(image, path):
    """保存带有中文路径的图片"""
    try:
        # 临时路径（避免中文路径问题）
        temp_path = "temp_image.jpg"
        cv2.imwrite(temp_path, image)

        # 将临时文件移动到目标路径
        shutil.move(temp_path, path)
        print(f"图片保存成功：{path}")
    except Exception as e:
        print(f"保存图片时出错：{e}")

def rotate_xml_and_draw(input_xml, output_xml, angle):
    """旋轉 XML 並同步旋轉圖片"""
    if not os.path.exists('rotated_images'):
        os.makedirs('rotated_images')
    # 解析 XML 文件
    tree = ET.parse(input_xml)
    root = tree.getroot()

    for image_elem in root.findall("image"):
        # 提取圖片相關信息
        image_path = image_elem.get("name")
        width = int(image_elem.get("width"))
        height = int(image_elem.get("height"))
        cx, cy = width / 2, height / 2  # 旋轉中心
        
        # 加載圖片
        image_path_full = os.path.join(r'C:/Users/TUF/Desktop/Irene/ML/Project/images/', image_path)
        image = load_image_with_chinese_path(image_path_full)
        if image is None:
            print(f"無法加載圖片：{image_path}")
            continue

        # 旋轉圖片
        rotated_image, rotation_matrix = rotate_image(image, angle)

        for polygon in image_elem.findall("polygon"):
            # 解析原始多邊形座標
            points_str = polygon.get("points")
            points = parse_polygon_points(points_str)
            
            # 旋轉多邊形
            rotated_points = []
            for x, y in points:
                point = np.array([[x, y, 1]]).T
                rotated_point = np.dot(rotation_matrix, point).flatten()
                rotated_points.append((rotated_point[0], rotated_point[1]))
            
            # 寫回 XML
            polygon.set("points", format_polygon_points(rotated_points))
            
            # 繪製旋轉後的多邊形
            rotated_np = np.array(rotated_points, np.int32).reshape((-1, 1, 2))
            # cv2.polylines(rotated_image, [rotated_np], isClosed=True, color=(0, 255, 0), thickness=2)

        # 保存顯示圖片結果
        image_name_utf8 = os.path.basename(image_path)  # 获取图片的原始文件名（含中文）
        rotated_filename = f"rotated_{image_name_utf8}"
        output_image_path = os.path.join('rotated_images', rotated_filename)
        save_image_with_chinese_name(rotated_image, output_image_path)
        cv2.imwrite(output_image_path, rotated_image)
        print(f"旋轉後的圖片已保存到 {output_image_path}")

        # 更新 XML 中的 image name
        image_elem.set("name", rotated_filename)

    # 保存旋轉後的 XML
    tree.write(output_xml, encoding='utf-8', xml_declaration=True)
    print(f"旋轉後的 XML 已保存到 {output_xml}")

# 使用範例
input_xml = "xml/output.xml"            # 原始 XML 文件
output_xml = "rotated_output.xml"  # 保存旋轉後的 XML 文件
angle = 45                         # 旋轉角度

rotate_xml_and_draw(input_xml, output_xml, angle)
