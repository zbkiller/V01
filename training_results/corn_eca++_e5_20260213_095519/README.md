# Training Results: corn_eca++_e5_20260213_095519

## Model Configuration
- **Model**: yolov8-eca++.yaml
- **Description**: Custom attention mechanism
- **Size**: YOLOv8n (medium variant - 25.9M parameters)
- **Dataset**: plantdoc v4
- **Epochs**: 5
- **Batch Size**: 2
- **Image Size**: 640
- **Training Time**: 10.18 minutes (0.17 hours)
- **Timestamp**: 20260213_095519

## Results
- **mAP50**: 0.0389
- **mAP50-95**: 0.0168
- **Precision**: 0.8804
- **Recall**: 0.1580
- **Best Epoch**: 5/5

## Validation
- **Val mAP50**: 0.0378
- **Val mAP50-95**: 0.0158
- **Val Precision**: 0.3275
- **Val Recall**: 0.0595

## Files
- Best Model: `training_results/corn_eca++_e5_20260213_095519/weights/best.pt`
- Last Model: `training_results/corn_eca++_e5_20260213_095519/weights/last.pt`
- Training Curves: `training_results/corn_eca++_e5_20260213_095519/results.png`
- Confusion Matrix: `training_results/corn_eca++_e5_20260213_095519/confusion_matrix.png`

## Download
```bash
wget https://github.com/zbkiller/V01/raw/main/training_results/corn_eca++_e5_20260213_095519/weights/best.pt
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
- YOLOv8m with eca++
- Custom attention mechanism
- Trained for 5 epochs
- Ready for deployment
