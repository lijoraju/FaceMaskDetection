# Face Mask Detection

This project implements a face mask detection system using PyTorch, OpenCV, and MediaPipe. It can be used for real-time detection from a camera or for classifying images.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
  - [Real-time Detection](#real-time-detection)
- [Dataset](#dataset)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The COVID-19 pandemic highlighted the importance of wearing face masks to prevent the spread of respiratory illnesses. This project aims to automatically detect whether individuals are wearing face masks correctly using computer vision techniques.  It utilizes a Convolutional Neural Network (CNN) trained on a dataset of images with and without masks.

## Installation

1. **Clone the repository:**

  ```bash
  git clone https://github.com/lijoraju/FaceMaskDetection.git
  cd face-mask-detection
 ```

2. **Create a Conda environment (recommended):**

```bash
conda create --name myenv python=3.11
conda activate myenv
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Usage

### Training 

To train the model, run the following command:

```bash
python main.py --train 
```

### Inference

To perform inference on a single image, use the following command:

```bash
python main.py --inference path/to/your/image.jpg
```

### Real-time Detection

To run real-time face mask detection from your camera, execute the following command:

```bash
    python main.py --realtime
```

Press 'q' to quit the real-time detection.

## Dataset

The project uses a dataset of images with and without face masks from kaggle https://www.kaggle.com/datasets/andrewmvd/face-mask-detection.  You can specify the paths to your training and validation data in the config.yaml file.  Please make sure that the data is structured appropriately.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

MIT License
