# Training Results: corn_gam5-c2f_e250_20260212_100428

## Model Configuration
- **Model**: yolov8-gam5-c2f.yaml
- **Description**: Custom attention mechanism
- **Size**: YOLOv8x (medium variant - 25.9M parameters)
- **Dataset**: plantdoc v4
- **Epochs**: 250
- **Batch Size**: 8
- **Image Size**: 100
- **Training Time**: 67.76 minutes (1.13 hours)
- **Timestamp**: 20260212_100428

## Results
- **mAP50**: 0.2713
- **mAP50-95**: 0.1858
- **Precision**: 0.9291
- **Recall**: 0.6391
- **Best Epoch**: 82/250

## Validation
- **Val mAP50**: 0.2713
- **Val mAP50-95**: 0.1865
- **Val Precision**: 0.3059
- **Val Recall**: 0.3331

## Files
- Best Model: `training_results/corn_gam5-c2f_e250_20260212_100428/weights/best.pt`
- Last Model: `training_results/corn_gam5-c2f_e250_20260212_100428/weights/last.pt`
- Training Curves: `training_results/corn_gam5-c2f_e250_20260212_100428/results.png`
- Confusion Matrix: `training_results/corn_gam5-c2f_e250_20260212_100428/confusion_matrix.png`

## Download
```bash
wget https://github.com/zbkiller/V01/raw/main/training_results/corn_gam5-c2f_e250_20260212_100428/weights/best.pt
```

## Use Model
```python
from ultralytics import YOLO

# Load model
model = YOLO('best.pt')

# Predict
results = model('corn_leaf_image.jpg')
results[0].show()

# Export
model.export(format='onnx')  # ONNX
model.export(format='engine')  # TensorRT
model.export(format='tflite')  # TFLite
```

## Performance
- YOLOv8m with gam5-c2f
- Custom attention mechanism
- Trained for 250 epochs
- Ready for deployment
