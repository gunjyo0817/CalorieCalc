Since there is not only one member doing preprocess, we need some files to rapidly merge our work, or fix some conflicts.
1. fix_xml_label.py: correct wrong labels to correct ones within whole xml files
2. label_high_class.py: add class 36 from origin_labeld_txts
3. merge_2xml_to1.pt: merge the label list and ids of 2 xml files
4. sementation_by_cv2.py: transform bounding boxes into polygon when labeling
5. xml_to_yolo_txt.py: transform xml files into txt files for yolo training
