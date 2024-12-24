# import os
# import cv2
# import shutil
# from ultralytics import YOLO

# def clear_directory(directory):
#     if os.path.exists(directory):
#         for filename in os.listdir(directory):
#             file_path = os.path.join(directory, filename)
#             try:
#                 if os.path.isfile(file_path) or os.path.islink(file_path):
#                     os.unlink(file_path)
#                 elif os.path.isdir(file_path):
#                     shutil.rmtree(file_path)
#             except Exception as e:
#                 print(f'Failed to delete {file_path}. Reason: {e}')

# def predict_images():
#     model_path = 'runs/detect/train3/weights/best.pt'  # Path to the best model weights
#     images_dir = 'test/images'  # Directory containing images for prediction
#     output_dir = 'results'  # Directory to save the results
#     confidence_threshold = 0.6

#     # Clear the output directory
#     clear_directory(output_dir)

#     # Load the trained model
#     model = YOLO(model_path)

#     # Create the output directory if it does not exist
#     os.makedirs(output_dir, exist_ok=True)

#     # Iterate over all images in the images directory
#     for image_name in os.listdir(images_dir):
#         # Full path to the image file
#         image_path = os.path.join(images_dir, image_name)

#         # Read the image using OpenCV
#         image = cv2.imread(image_path)
#         if image is None:
#             print(f"Failed to read image {image_path}")
#             continue

#         # Perform inference on the image
#         results = model(image)

#         # Get image dimensions
#         img_height, img_width, _ = image.shape

#         highest_confidence_boxes = {"pen": None, "cap": None}
#         highest_confidence_scores = {"pen": -1, "cap": -1}
        
#         for result in results:
#             # Filter boxes by confidence threshold
#             for box in result.boxes:
#                 class_id = int(box.cls[0])
#                 class_name = model.names[class_id]
#                 print("find~~~~~~~~~~~~~~~~~~~~~~~~~~~")
#                 if class_name in highest_confidence_boxes:
#                     print("class name in highest confidence")
#                     if box.conf[0] >= confidence_threshold and box.conf[0] > highest_confidence_scores[class_name]:
#                         print("confident!!!!!!!!!!!!!")
                        
#                         highest_confidence_scores[class_name] = box.conf[0]
#                         highest_confidence_boxes[class_name] = box


#         # Process each result in the results list
#         for result in results:
#             # Filter boxes by confidence threshold
#             valid_boxes = [box for box in result.boxes if box.conf[0] >= confidence_threshold]

#             if not valid_boxes:
#                 print(f"No valid detections for image {image_name} with confidence threshold {confidence_threshold}")
#                 continue

#             # Plot valid boxes only
#             plotted_image = image.copy()

            
#             for box in valid_boxes:
#                 # Get bounding box coordinates
#                 x1, y1, x2, y2 = box.xyxy[0]
#                 x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#                 # Draw the bounding box
#                 cv2.rectangle(plotted_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
#                 # Put label and confidence
#                 label = f"{model.names[int(box.cls[0])]}: {box.conf[0]:.2f}"
#                 cv2.putText(plotted_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

#                 # Print bounding box information
#                 class_id = int(box.cls[0])  # class id
#                 confidence = box.conf[0]  # confidence score
#                 width = x2 - x1
#                 height = y2 - y1
#                 center_x = (x1 + x2) / 2
#                 center_y = (y1 + y2) / 2
#                 center_x_ratio = center_x / img_width
#                 center_y_ratio = center_y / img_height
#                 width_ratio = width / img_width
#                 height_ratio = height / img_height
#                 width_height_ratio = width / height
#                 print(f"Class: {model.names[class_id]}, Confidence: {confidence:.2f}, BBox: ({x1}, {y1}, {width}, {height}), Center Ratio: ({center_x_ratio:.2f}, {center_y_ratio:.2f})")
#                 print(f"Width Ratio: {width_ratio:.2f}, Height Ratio: {height_ratio:.2f}, Width / Height: {width_height_ratio:.2f}")

#             # Show the image with bounding boxes
#             # cv2.imshow('Prediction', plotted_image)
#             # cv2.waitKey(0)  # Wait for a key press to display the next image
#             # cv2.destroyAllWindows()  # Close the window

#             # Save the result image
#             output_image_path = os.path.join(output_dir, image_name)
#             cv2.imwrite(output_image_path, plotted_image)
#             print(f"Saved results to {output_image_path}")

# if __name__ == '__main__':
#     predict_images()

import os
import cv2
import shutil
from ultralytics import YOLO

def clear_directory(directory):
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

def predict_images():
    model_path = 'C:\\Users\\TUF\\Desktop\\Irene\\ML\\Project_test_change1_only_183\\runs\\detect\\train4\\weights\\best.pt'  # Path to the best model weights
    images_dir = 'C:\\Users\\TUF\\Desktop\\Irene\\ML\\Project_test_change1_only_183\\test\\images'  # Directory containing images for prediction
    output_dir = 'results'  # Directory to save the results
    confidence_threshold = 0.4

    # Clear the output directory
    clear_directory(output_dir)

    # Load the trained model
    model = YOLO(model_path)

    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over all images in the images directory
    for image_name in os.listdir(images_dir):
        # Full path to the image file
        image_path = os.path.join(images_dir, image_name)

        # Read the image using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read image {image_path}")
            continue

        # Perform inference on the image
        results = model(image)

        # Get image dimensions
        img_height, img_width, _ = image.shape

        # YOLO format result file
        yolo_txt_path = os.path.join(output_dir, os.path.splitext(image_name)[0] + ".txt")
        with open(yolo_txt_path, 'w') as yolo_file:

            for result in results:
                # Filter boxes by confidence threshold
                valid_boxes = [box for box in result.boxes if box.conf[0] >= confidence_threshold]

                if not valid_boxes:
                    print(f"No valid detections for image {image_name} with confidence threshold {confidence_threshold}")
                    continue

                # Plot valid boxes only
                plotted_image = image.copy()

                for box in valid_boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Draw the bounding box
                    cv2.rectangle(plotted_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    
                    # Put label and confidence
                    label = f"{model.names[int(box.cls[0])]}: {box.conf[0]:.2f}"
                    cv2.putText(plotted_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                    # Calculate YOLO format values
                    class_id = int(box.cls[0])  # class id
                    confidence = box.conf[0]  # confidence score
                    width = x2 - x1
                    height = y2 - y1
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    center_x_ratio = center_x / img_width
                    center_y_ratio = center_y / img_height
                    width_ratio = width / img_width
                    height_ratio = height / img_height

                    # Write YOLO format to .txt file
                    yolo_file.write(f"{class_id} {center_x_ratio:.6f} {center_y_ratio:.6f} {width_ratio:.6f} {height_ratio:.6f}\n")

                    print(f"Class: {model.names[class_id]}, Confidence: {confidence:.2f}, YOLO: ({class_id}, {center_x_ratio:.6f}, {center_y_ratio:.6f}, {width_ratio:.6f}, {height_ratio:.6f})")

                # Save the result image
                output_image_path = os.path.join(output_dir, image_name)
                cv2.imwrite(output_image_path, plotted_image)
                print(f"Saved results to {output_image_path}")
                print(f"Saved YOLO annotations to {yolo_txt_path}")

if __name__ == '__main__':
    predict_images()
