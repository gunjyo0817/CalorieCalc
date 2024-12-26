import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np

food_labels = [
    "cucumber", "chicken", "shrimp", "brocolli", "bean", "salmon", "pork", "rice", 
    "pumpkin", "egg_dish", "nut", "carrot", "king_oyster", "vegetable", 
    "winter_melon", "mango", "avocado", "baby_corn", "beef", "sprout", 
    "eggplant", "mushroom", "potato", "brocoli", "corn", "toast", "onion", 
    "cabbage", "egg", "green_pepper", "green_bean", "tofu", "fish", "tomato", 
    "lemon", "yam", "high"
]
food_kcal_per_ml = {
    "beef": 257.5,
    "carrot": 26.24,
    "shrimp": 103.95,
    "cucumber": 15.2,
    "onion": 37.6,
    "tomato": 17.1,
    "eggplant": 23.5,
    "sprout": 27,
    "baby_corn": 23.92,
    "egg_dish": 155,
    "cabbage": 23,
    "potato": 73.92,
    "chicken": 250.95,
    "salmon": 212.16,
    "mushroom": 20.68,
    "fish": 206,
    "winter_melon": 12.48,
    "corn": 82.56,
    "tofu": 73.72,
    "brocoli": 30.94,
    "green_bean": 29.14,
    "avocado": 148.8,
    "mango": 55.8,
    "vegetable": 18,
    "yam": 112.1,
    "pumpkin": 112.1,
    "green_pepper": 18.6,
    "toast": 265,
    "pork": 249.26,
    "egg": 155,
    "bean": 329.65,
    "nut": 607
}

filename_to_calorie = {
    '水煮牛肉': 550,
    '來自挪威的烤薄鹽鯖魚': 530,
    '不用客氣滿滿肉片': 520,
    '靈感來自古宇利島的蝦蝦飯': 430,
    '清蒸義式雞腿': 580,
    '義式水煮雞胸肉': 430,
    '秘製岳母牛腱肉': 470,
    '香滷雞腿': 601.7,
    '水煮嫩雞胸': 545.6,
    '穀飼牛漢堡排': 591,
    '泡菜豬里肌': 468.9,
    '鹽烤鯖魚': 542.3,
    '韓式菇菇雞': 526.3,
    '麻香口水雞': 578.5,
    '日式壽喜牛便當': 686.8,
    '日式蔥鹽雞': 510.7,
    '泰式打拋雞': 505,
    '蒜泥豬里肌': 470.1,
    '清滷牛腱': 526.5,
    '照燒雞腿排': 632.3,
    '鷹嘴豆咖哩': 623.9,
    '菜飯餐盒': 350,
    '00': 492,
    '01': 443,
    '02': 478,
    '03': 341,
    '04': 405,
    '05': 429,
    '06': 386,
    '07': 415,
    '08': 252.4,
    '09': 483,
    '711繽紛鮮蔬烤雞便當_1': 371,
    '711繽紛鮮蔬烤雞便當_2': 371,
    '711繽紛鮮蔬烤雞便當_3': 353,
    '義式雞腿排餐盒': 572.5,
    '鹽烤鮭魚菲力餐盒': 587.3,
    '舒肥雞胸肉餐盒': 623.4,
    '日式煎豆腐餐盒': 553.6,
    '舒肥牛排餐盒': 515.3,
    '香檸松阪豬餐盒': 516.1,
    '蒜烤霜降豬餐盒': 581.6,
    '全家烤多蛋白餐盒_1': 463,
    '全家烤多蛋白餐盒_2': 463,
    '全家健身G肉餐盒': 461,
    '全家美式雞丁餐盒': 466,
    '初夏牛腱沙律1': 552.1,
    '嫩滑雞胸2': 449.8,
    '叉雞飯3': 485.5,
    '煎牛扒蘆筍4': 455.7,
    '鯛魚13': 685,
    '鯛魚3': 685,
    '雞腿26': 694,
    '雞腿24': 694,
    '雞腿17': 694,
    '雞腿4': 694,
    '雞排23': 751,
    '雞排20': 751,
    '雞排18': 751,
    '雞排14': 751,
    '雞胸雙拼38': 628,
    '蔥雞胸30': 696,
    '蔥雞胸27': 696,
    '蔥雞胸14': 696,
    '蔥雞胸13': 696,
    '蒲燒鯛魚12': 685,
    '嫩雞胸25': 646,
    '嫩雞胸2': 646,
    '義式杏鮑菇39': 524,
    '飯1': 525,
    '舒肥原味雞胸40': 579,
    '舒肥原味雞胸31': 579,
    '舒肥原味雞33': 579,
    '排骨6': 670,
    '芥末椒鹽大鮭魚41': 728,
    '低脂雙拼32': 628,
    '西班牙煙燻雞腿35': 739,
    '西班牙煙燻雞43': 739,
    '西班牙煙燻去骨雞44': 739,
    '西班牙煙燻34': 739,
    '西班牙去古蹟37': 739,
    '地瓜減脂36': 166,
    '白煮魚15': 577,
    '白煮魚5': 577,
    '白身魚16': 577,
    '打拋豬22': 780,
    '打拋豬21': 780,
    '打拋豬11': 780,
    '打拋豬10': 780,
    '牛腱9': 685,
    '牛腱8': 685,
    '牛腱7': 685,
    '牛腱6': 685,
    '711松阪豬花椰菜減醣飯': 276,
    '牛肉餐02': 460,
    '里肌豬餐01': 459,
    '里肌豬餐02': 459,
    '里肌豬餐03': 459,
    '里肌豬餐05': 459,
    '雞胸餐01': 400.5,
    '雞腿餐02': 493.7,
    '雞腿餐04': 493.7,
    '味噌松阪豬': 654,
    '泰式烤雞腿': 743,
    '02算來算去涮涮牛': 582,
    '03如魚得水水煮魚': 529,
    '05力爭上游鮮鮭魚': 588,
    '07浩克小肌雞胸': 556,
    '08清新脫俗素餐盒': 384,
    "醬煮牛肉425": 425,
    "戰斧鮭魚614": 614,
    "蒜泥豬肉477": 477,
    "嫩雞胸548": 548,
    "嫩煎雞腿排597": 597,
    "舒肥雞胸413": 413,
    "舒肥雞胸413_3": 413,
    "舒肥雞胸413_2": 413,
    "手撕雞573": 573,
    "巴沙魚433": 433,
    "711香烤雞胸鮮蔬餐": 355,
    "牛肉餐01": 460,
    "牛肉餐03": 460,
    "里肌豬餐04": 459,
    "雞胸餐02": 400.5,
    "雞胸餐03": 400.5,
    "雞腿餐01": 493.7,
    "雞腿餐03": 493.7,
    "01舒肥大肌雞胸肉": 534,
    "04台灣豬就是讚": 652,
    "beef": 257.5,
    "carrot": 26.24,
    "shrimp": 103.95,
    "cucumber": 15.2,
    "onion": 37.6,
    "broccoli": 30.94,
    "brocolli": 30.94,
    "tomato": 17.1,
    "eggplant": 23.5,
    "chinese broccoli": 19.8,
    "scallions": 28.48,
    "zucchini": 15.98,
    "king_oyster": 33.25,
    "squash": 24.18,
    "sprout": 27,
    "baby_corn": 23.92,
    "egg_dish": 155,
    "radish": 15.04,
    "cabbage": 23,
    "potato": 73.92,
    "beansprouts": 27.3,
    "chicken": 250.95,
    "salmon": 212.16,
    "mushroom": 20.68,
    "asparagus": 18.2,
    "baby corn": 23.92,
    "garlic": 141.55,
    "fish": 206,
    "winter_melon": 12.48,
    "corn": 82.56,
    "egg tofu": 82.32,
    "tofu": 73.72,
    "edamame": 114.95,
    "brocoli": 30.94,
    "green_bean": 29.14,
    "avocado": 148.8,
    "mango": 55.8,
    "vegetable": 18,
    "yam": 112.1,
    "pumpkin": 112.1,
    "green_pepper": 18.6,
    "lemon": 27.26,
    "purple rice": 365,
    "toast": 265,
    "sweet potato": 81.7,
    "rice": 130,
    "pork": 249.26,
    "egg": 155,
    "cherry tomatoes": 16.92,
    "bean": 329.65,
    "nut": 607
}
test_names = [
    "蒜泥豬肉477",
    "巴沙魚433",
    "牛肉餐01",
    "牛肉餐03",
    "里肌豬餐04",
    "雞胸餐02",
    "雞腿餐01",
    "雞腿餐03",
    "01舒肥大肌雞胸肉",
    "04台灣豬就是讚",
    "不用客氣滿滿肉片",
    "來自挪威的烤薄鹽鯖魚",
    "水煮嫩雞胸",
    "清滷牛腱",
    "711繽紛鮮蔬烤雞便當_1",
    "日式煎豆腐餐盒",
    "舒肥原味雞胸31",
    "牛腱8",
    "里肌豬餐02",
    "08清新脫俗素餐盒"
]
food_nutrients = {
    'cucumber': {'protein': 0.65, 'fat': 0.11, 'carbs': 3.63},
    'chicken': {'protein': 31.0, 'fat': 3.6, 'carbs': 0},
    'shrimp': {'protein': 24.0, 'fat': 0.3, 'carbs': 0},
    
    'bean': {'protein': 21.0, 'fat': 0.8, 'carbs': 35.0},
    'salmon': {'protein': 22.0, 'fat': 12.0, 'carbs': 0},
    'pork': {'protein': 25.0, 'fat': 14.0, 'carbs': 0},
    
    'pumpkin': {'protein': 1.0, 'fat': 0.1, 'carbs': 7.0},
    'egg_dish': {'protein': 9.0, 'fat': 5.0, 'carbs': 1.0},
    'nut': {'protein': 20.0, 'fat': 50.0, 'carbs': 20.0},
    'carrot': {'protein': 0.9, 'fat': 0.2, 'carbs': 9.6},
    'king_oyster': {'protein': 3.5, 'fat': 0.3, 'carbs': 5.0},
    'vegetable': {'protein': 1.0, 'fat': 0.1, 'carbs': 3.0},
    'winter_melon': {'protein': 0.6, 'fat': 0.1, 'carbs': 4.2},
    'mango': {'protein': 0.8, 'fat': 0.4, 'carbs': 24.7},
    'avocado': {'protein': 2.0, 'fat': 15.0, 'carbs': 9.0},
    'baby_corn': {'protein': 2.4, 'fat': 0.2, 'carbs': 19.0},
    'beef': {'protein': 26.0, 'fat': 15.0, 'carbs': 0},
    'sprout': {'protein': 3.0, 'fat': 0.1, 'carbs': 8.0},
    'eggplant': {'protein': 1.0, 'fat': 0.2, 'carbs': 5.9},
    'mushroom': {'protein': 3.0, 'fat': 0.3, 'carbs': 4.0},
    'potato': {'protein': 2.0, 'fat': 0.1, 'carbs': 17.0},
    'corn': {'protein': 3.3, 'fat': 1.2, 'carbs': 19.0},
    'toast': {'protein': 6.0, 'fat': 1.0, 'carbs': 12.0},
    'onion': {'protein': 1.1, 'fat': 0.1, 'carbs': 9.3},
    'cabbage': {'protein': 1.3, 'fat': 0.1, 'carbs': 5.8},
    'egg': {'protein': 6.0, 'fat': 5.0, 'carbs': 0.6},
    'green_pepper': {'protein': 0.9, 'fat': 0.2, 'carbs': 6.0},
    'green_bean': {'protein': 2.0, 'fat': 0.2, 'carbs': 7.0},
    'tofu': {'protein': 8.0, 'fat': 4.0, 'carbs': 2.0},
    'fish': {'protein': 20.0, 'fat': 5.0, 'carbs': 0},
    'tomato': {'protein': 0.9, 'fat': 0.2, 'carbs': 4.0},
    
    'yam': {'protein': 1.5, 'fat': 0.1, 'carbs': 27.0}
}


def calculate_polygon_area(points):
    """Compute the area of a polygon using the Shoelace formula."""
    n = len(points)
    if n < 3:
        return 0
    area = 0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - y1 * x2
    return abs(area) / 2

def parse_xml_to_dataframe_with_ratios(xml_file):
    """Parse an XML file and return a DataFrame with label ratios."""
    tree = ET.parse(xml_file)
    root = tree.getroot()

    image_names = []
    areas = []

    for image in root.findall('image'):
        image_name = image.get('name')
        print(image_name)
        label_areas = {label: 0 for label in food_labels} 
        total_area = 0  
        label_areas["high"] = 0

        for polygon in image.findall('polygon'):
            label = polygon.get('label')
            if label not in food_labels:
                continue 
            points = polygon.get('points')
            points = [tuple(map(float, point.split(','))) for point in points.split(';')]
            area = calculate_polygon_area(points)
            label_areas[label] += area
            
            if(label != "high"): total_area += area
          
        if total_area > 0:
            label_ratios = {label: label_areas[label] / total_area for label in food_labels}
        else:
            label_ratios = {label: 0 for label in food_labels}  

        image_names.append(image_name)
        areas.append([label_ratios[label] for label in food_labels])

    df = pd.DataFrame(areas, columns=food_labels)
    df.insert(0, 'image_name', image_names)

    df['image_name'] = df['image_name'].apply(lambda x: x.split('.')[0]) 
    df['calories'] = df['image_name'].map(filename_to_calorie)  

    df['protein'] = 0
    df['fat'] = 0
    df['carbs'] = 0

      
    for food, nutrients in food_nutrients.items():
            if food in ["salmon", "pumpkin", "rice", "king_oyster", "potato", "brocolli"]: continue;
            if food in df.columns:  
                df['protein'] += df[food] * nutrients['protein']
                df['fat'] += df[food] * nutrients['fat']
                df['carbs'] += df[food] * nutrients['carbs']
    df['GMM'] = np.log(df['carbs'] * df['fat'] * df['protein'] + 1e-5)/np.log(3);
    
    return df
    

xml_file = 'test.xml'
df = parse_xml_to_dataframe_with_ratios(xml_file)

print(df.head())
csv_file = 'test.csv'
df.to_csv(csv_file, index=False, encoding='utf-8-sig')