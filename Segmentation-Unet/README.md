# Segmentation - Unet

The segmentation process is used to accurately determine the area of each food item, enabling precise prediction results.

This project is based on the [segmentation_models](https://github.com/qubvel/segmentation_models) GitHub repository.

## Requirements

Ensure you have all necessary dependencies installed by running the following command. Using a virtual environment is recommended:

```bash
pip install -r requirements.txt
```

## Generating Training Mask

Run the following command to generate training masks from the xml file.
```
python3 generate_train_masks.py
```
Notice that the output directory and the XML file path can be customized in the code to suit your requirements.

## Training the Model

To train the model, execute the training script with the following command:

```
CUDA_VISIBLE_DEVICES="" python3 YOLO_binary_seg.py
```

Note: To avoid complications with CUDA version compatibility, you can use `CUDA_VISIBLE_DEVICES=""`.

### Arguments:
- `--img_width`: Width of input images (pixels). Default: 256
- `--img_height`: Height of input images (pixels). Default: 256
- `--epochs`: Number of epochs to train the model. Default: 50
- `--batch_size`: Batch size for training. Default: 16
- `--backbone`: Backbone for the U-Net model. Default: resnet34. For more backbones, please refer to the [segmentation_models](https://github.com/qubvel/segmentation_models) github repository.

The model's weight will be saved in the `./weight` directory by default.

### Example command:
```
CUDA_VISIBLE_DEVICES="" python3 YOLO_binary_seg.py --batch_size 32 --backbone resnet50
```

## Predicting with the Model

After training, use the model for predictions by running:

```
CUDA_VISIBLE_DEVICES="" python3 YOLO_predict.py
```

After predicting, it will generate a xml file for predicting calorie, and png files for visualizing the segmented pictures.

The png results will be saved to `./output/v2_test`

The xml file will be saved to `./output/xml_files`

Notice that the output directory can be customized in the code to suit your requirements.



## Evaluating Accuracy

To evaluate the model's accuracy, use the evaluation script:

### Example:

```
python3 YOLO_accuracy.py
```

It will show the overall mean IoU and pixel accuracy.