# Hand Gesture Recognition with Deep Learning

A real-time hand gesture recognition system using deep learning and computer vision. This project recognizes three hand gestures: closed fist, open hand, and peace sign using a CNN or transfer learning model (MobileNetV2).

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Features

- **Real-time gesture recognition** using webcam
- **Automatic hand tracking** with skin detection
- **Transfer learning** support with MobileNetV2
- **Custom CNN** architecture option
- **Region of Interest (ROI)** tracking for improved accuracy
- **Prediction smoothing** for stable results
- **Interactive controls** for live parameter adjustment
- **Visual feedback** with confidence scores and probability bars
- **Debug mode** for skin detection visualization

## Demo

The system recognizes three gestures:
- ✊ **Closed Fist**
- 🖐️ **Open Hand**
- ✌️ **Peace Sign**

## Requirements

```
tensorflow>=2.8.0
opencv-python>=4.5.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/hand-gesture-recognition.git
cd hand-gesture-recognition
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare your dataset:
```
dataset/
├── closed_fist/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── open_hand/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
└── peace_sign/
    ├── img1.jpg
    ├── img2.jpg
    └── ...
```

## Usage

### Training the Model

Run the training script to train a new model:

```bash
python train.py
```

**Training options:**
- Set `use_transfer_learning = True` for MobileNetV2 transfer learning (recommended)
- Set `use_transfer_learning = False` for custom CNN architecture
- Adjust `img_size`, `batch_size`, and `epochs` as needed

The script will:
- Train the model with data augmentation
- Apply class weights for imbalanced datasets
- Use callbacks for learning rate reduction and early stopping
- Save the best model as `best_model.h5`
- Generate confusion matrix and training history plots

### Real-time Detection

Run the detection script to start recognizing gestures:

```bash
python detect.py
```

Make sure `MODEL_PATH` points to your trained model file.

## Interactive Controls

During real-time detection, use these keyboard shortcuts:

| Key | Action |
|-----|--------|
| `q` | Quit the application |
| `s` | Toggle prediction smoothing |
| `r` | Reset smoothing buffer and hand position |
| `o` | Toggle Region of Interest (ROI) |
| `t` | Toggle automatic hand tracking |
| `d` | Show/hide skin detection debug view |
| `[` | Decrease ROI size |
| `]` | Increase ROI size |
| `+` / `=` | Increase confidence threshold |
| `-` / `_` | Decrease confidence threshold |

## Configuration

### Detection Settings (`detect.py`)

```python
# Model configuration
MODEL_PATH = "best_model_v2.h5"
img_size = 128
confidence_threshold = 0.50

# Smoothing
USE_SMOOTHING = True
SMOOTHING_FRAMES = 5

# ROI settings
USE_ROI = True
ROI_SIZE = 300
TRACK_HAND = True

# Hand detection constraints
MIN_HAND_AREA = 5000
MAX_HAND_AREA = 50000
```

### Training Settings (`train.py`)

```python
img_size = 128
batch_size = 32
epochs = 50
use_transfer_learning = True
num_classes = 3
```

## Model Architecture

### Transfer Learning (MobileNetV2)
- Base: MobileNetV2 pretrained on ImageNet
- Global Average Pooling
- Dense layers: 256 → 128 → num_classes
- Batch Normalization and Dropout for regularization
- Two-phase training: frozen base → fine-tuning

### Custom CNN
- 4 convolutional blocks with increasing filters (32 → 64 → 128 → 256)
- Batch Normalization after each conv layer
- MaxPooling and Dropout for regularization
- Global Average Pooling
- Dense layers: 256 → 128 → num_classes

## Performance Tips

1. **Lighting**: Ensure good, even lighting for better hand detection
2. **Background**: Use a contrasting background to your skin tone
3. **Distance**: Keep your hand at a comfortable distance from the camera
4. **ROI Tracking**: Enable hand tracking (`t` key) for automatic ROI positioning
5. **Smoothing**: Keep smoothing enabled for stable predictions
6. **Confidence Threshold**: Adjust threshold based on your accuracy requirements

## Project Structure

```
hand-gesture-recognition/
├── train.py                 # Model training script
├── detect.py                # Real-time detection script
├── dataset/                 # Training data directory
│   ├── closed_fist/
│   ├── open_hand/
│   └── peace_sign/
├── best_model.h5           # Saved model weights
├── confusion_matrix.png    # Model evaluation results
├── training_history.png    # Training metrics visualization
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Troubleshooting

**Model not detecting gestures:**
- Check lighting conditions
- Adjust `confidence_threshold`
- Try toggling ROI or hand tracking
- Verify model path is correct
- For open hand, spread out your fingers.

**Hand tracking not working:**
- Adjust skin detection HSV ranges in `detect.py`
- Check `MIN_HAND_AREA` and `MAX_HAND_AREA` settings
- Use debug mode (`d` key) to visualize skin detection

**Low accuracy:**
- Collect more training data
- Increase training epochs
- Try transfer learning if using custom CNN
- Adjust data augmentation parameters

## Future Improvements

- [ ] Add more gesture classes
- [ ] Implement gesture sequence recognition
- [ ] Add support for multiple hands
- [ ] Create a gesture-controlled application demo
- [ ] Add MediaPipe integration for improved hand tracking
- [ ] Support for custom gesture training via GUI

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Contributors

1.  Au-Revoir5
    https://github.com/Au-Revoir5/
2.  Sedoxus
    https://github.com/Sedoxus/

3.  HmmmAha
    https://github.com/HmmmAha/

4.  farrelpit
    https://github.com/farrelpit/

## Acknowledgments

- MobileNetV2 architecture from TensorFlow/Keras
- OpenCV for computer vision capabilities
- The open-source community for inspiration and tools

⭐ If you found this project helpful, please give it a star!

Made with Python