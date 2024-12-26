import xml.etree.ElementTree as ET
import os

def parse_xml_to_yolo(xml_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 动态生成 label_map
    label_map = generate_label_map(xml_folder)

    for xml_file in os.listdir(xml_folder):
        if not xml_file.endswith('.xml'):
            continue

        tree = ET.parse(os.path.join(xml_folder, xml_file))
        root = tree.getroot()

        # 遍历每个 <image> 节点
        for image in root.findall("image"):
            img_name = image.attrib["name"]
            img_width = int(image.attrib["width"])
            img_height = int(image.attrib["height"])

            # YOLO 格式输出文件
            yolo_file_path = os.path.join(output_folder, os.path.splitext(img_name)[0] + ".txt")

            with open(yolo_file_path, 'w') as yolo_file:
                # 遍历该图片的所有标注
                for polygon in image.findall("polygon"):
                    label = polygon.attrib["label"]
                    points = polygon.attrib["points"]

                    # 将多边形点转换为边界框
                    x_coords = [float(pt.split(',')[0]) for pt in points.split(';')]
                    y_coords = [float(pt.split(',')[1]) for pt in points.split(';')]

                    x_min = min(x_coords)
                    x_max = max(x_coords)
                    y_min = min(y_coords)
                    y_max = max(y_coords)

                    # 计算 YOLO 格式所需的中心点和宽高
                    x_center = (x_min + x_max) / 2 / img_width
                    y_center = (y_min + y_max) / 2 / img_height
                    width = (x_max - x_min) / img_width
                    height = (y_max - y_min) / img_height

                    # 获取类别索引
                    class_index = label_map[label]

                    # 写入 YOLO 格式：class_index x_center y_center width height
                    yolo_file.write(f"{class_index} {x_center} {y_center} {width} {height}\n")

def generate_label_map(xml_folder):
    """
    动态生成 label_map，根据所有 XML 文件中的标签。
    """
    label_set = set()

    for xml_file in os.listdir(xml_folder):
        if not xml_file.endswith('.xml'):
            continue

        tree = ET.parse(os.path.join(xml_folder, xml_file))
        root = tree.getroot()

        # 遍历所有 <polygon> 标签中的 label
        for polygon in root.findall(".//polygon"):
            label = polygon.attrib["label"]
            label_set.add(label)

    # 将标签按字母顺序排序并分配索引
    label_map = {label: idx for idx, label in enumerate(sorted(label_set))}
    # print("Generated label_map:", label_map)
    # for idx, label in enumerate(label_map):
    #     print(f"{idx}: {label}")
    label_map = {
        'cucumber': 0, 
        'chicken': 1,
        'shrimp': 2,
        'brocolli': 3,
        'bean': 4,
        'salmon': 5,
        'pork': 6,
        'rice': 7,
        'pumpkin': 8,
        'egg_dish': 9,
        'nut': 10,
        'carrot': 11,
        'king_oyster': 12,
        'vegetable': 13,
        'winter_melon': 14,
        'mango': 15,
        'avocado': 16,
        'baby_corn': 17,
        'beef': 18,
        'sprout': 19,
        'eggplant': 20,
        'mushroom': 21,
        'potato': 22,
        'brocoli': 23,
        'corn': 24,
        'toast': 25,
        'onion': 26,
        'cabbage': 27,
        'egg': 28,
        'green_pepper': 29,
        'green_bean': 30,
        'tofu': 31,
        'fish': 32,
        'tomato': 33,
        'lemon': 34,
        'yam': 35
    }
    return label_map

# 输入和输出路径
xml_folder = "xml"
output_folder = "labels"

parse_xml_to_yolo(xml_folder, output_folder)
