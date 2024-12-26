# Object Detection - YOLO
This repository provides a YOLO-based object detection model trained to identify food items in images of healthy lunch box.

## Testing the Model
To test the model on images, run ```test.py``` as follows:

```
python test.py
```

### Input:
- **The YOLO model trained weights**
    - Located at ```runs/detect/train4/weights/best.pt``` by default. (If your model is in a different location, update the model_path in the script)
- **Images to be processed**
    - Located in the ```test/images/``` by default (You can place your own images in this folder) 

### Output:

After running the script, the ```results/directory``` will contain:

- ```imageN.jpg``` : image with bounding boxes
- ```imageN.txt``` : YOLO-format annotation
    - in the format of ```class_id center_x center_y width height```

The output will then be passed to the **Segmentation model** for further processing.

### Notes:
- The script processes all images in the test/images/ directory. Ensure your images are in a compatible format (e.g., .jpg, .png).
- If you modify the model or input paths, make sure to update the script accordingly.
