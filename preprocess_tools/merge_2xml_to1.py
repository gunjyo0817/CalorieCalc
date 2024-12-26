import xml.etree.ElementTree as ET
from xml.dom import minidom

def merge_xml_files(input_file1, input_file2, output_file):
    """Merge two XML files by combining their <image> elements, assigning unique IDs starting from 0, 
    ensuring no duplicate <image> elements by checking the 'name' attribute, and adding a global <labels> section."""
    try:
        tree1 = ET.parse(input_file1)
        root1 = tree1.getroot()
        tree2 = ET.parse(input_file2)
        root2 = tree2.getroot()
        
        # Create the root element for the merged XML
        merged_root = ET.Element("annotations")

        # Create a global <labels> element with unique <label> entries
        labels = ET.SubElement(merged_root, "labels")

        # Use a set to collect unique labels
        label_set = set()

        # Use a set to track existing <image> names
        existing_names = set()

        # ID counter starting from 0
        image_id = 0

        # Append <image> elements from the first XML file and collect labels
        for image in root1.findall("image"):
            image_name = image.get("name")
            if image_name not in existing_names:
                existing_names.add(image_name)  # Add to the set of existing names
                for polygon in image.findall("polygon"):
                    label_set.add(polygon.get("label").lower());
                    polygon.set("label", polygon.get("label").lower());
                image.set("id", str(image_id))  # Set unique ID starting from 0
                merged_root.append(image)
                image_id += 1  # Increment ID counter
            else: print(f"already have {image_name}!")

        # Append <image> elements from the second XML file and collect labels
        for image in root2.findall("image"):
            image_name = image.get("name")
            if image_name not in existing_names:
                existing_names.add(image_name)  # Add to the set of existing names
                for polygon in image.findall("polygon"):
                    label_set.add(polygon.get("label").lower());
                    polygon.set("label", polygon.get("label").lower());
                image.set("id", str(image_id))  # Continue with the unique ID counter
                merged_root.append(image)
                image_id += 1  # Increment ID counter
            else: print(f"already have {image_name}!")

        print(label_set);
        # Create <label> elements for each unique label
        for label_name in label_set:
            label = ET.SubElement(labels, "label")
            name = ET.SubElement(label, "name")
            name.text = label_name

        # Convert ElementTree to a string
        rough_string = ET.tostring(merged_root, encoding="unicode")

        # Use minidom to prettify XML while avoiding extra blank lines
        reparsed = minidom.parseString(rough_string)
        pretty_xml = "\n".join([line for line in reparsed.toprettyxml(indent="  ").splitlines() if line.strip()])

        # Write the prettified XML to a file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(pretty_xml)

        print(f"Merged XML with labels and IDs starting from 0 saved to {output_file}")
    except Exception as e:
        print(f"Error during merging: {e}")

# 示例用法
input_file1 = "input_xmls/test1.xml"
input_file2 = "input_xmls/test2.xml"
output_file = "merged.xml"

merge_xml_files(input_file1, input_file2, output_file)