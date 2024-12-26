import xml.etree.ElementTree as ET

# 讀取 XML 文件
input_file = "xml/output.xml"  # 替換成你的 XML 文件名
output_file = "xml/output.xml"

tree = ET.parse(input_file)
root = tree.getroot()

# 遍歷所有 <image> 節點
for image in root.findall(".//image"):
    # 在 <image> 節點中找到所有 <polygon> 子節點
    for polygon in image.findall(".//polygon"):
        # 修改 label 屬性值
        if polygon.attrib.get("label") == "radish":
            polygon.set("label", "winter_melon")

# 將修改後的 XML 寫回文件
tree.write(output_file, encoding="utf-8", xml_declaration=True)

print(f"已將 'zucchini' 改為 'cucumber' 並儲存到 {output_file}")
