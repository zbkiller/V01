# Training Results: corn_hp_gam_e70_20260125_022614

## Model Configuration
- **Model**: yolov8-hp_gam.yaml
- **Description**: High Performance GAM (best)
- **Size**: YOLOv8m (medium variant - 25.9M parameters)
- **Dataset**: corn-leaf-disease-hgosu v17
- **Epochs**: 70
- **Batch Size**: 8
- **Image Size**: 640
- **Training Time**: 281.66 minutes (4.69 hours)
- **Timestamp**: 20260125_022614

## Results
- **mAP50**: 0.9643
- **mAP50-95**: 0.8851
- **Precision**: 0.9594
- **Recall**: 0.9262
- **Best Epoch**: 63/70

## Validation
- **Val mAP50**: 0.9628
- **Val mAP50-95**: 0.8860
- **Val Precision**: 0.9367
- **Val Recall**: 0.9209

## Files
- Best Model: `training_results/corn_hp_gam_e70_20260125_022614/weights/best.pt`
- Last Model: `training_results/corn_hp_gam_e70_20260125_022614/weights/last.pt`
- Training Curves: `training_results/corn_hp_gam_e70_20260125_022614/results.png`
- Confusion Matrix: `training_results/corn_hp_gam_e70_20260125_022614/confusion_matrix.png`

## Download
```bash
wget https://github.com/zbkiller/V01/raw/main/training_results/corn_hp_gam_e70_20260125_022614/weights/best.pt
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
- YOLOv8m with hp_gam
- High Performance GAM (best)
- Trained for 70 epochs
- Ready for deployment
