American Sign Language Recognition using Computer Vision

- Overview
This project focuses on recognizing American Sign Language (ASL) hand gestures using Computer Vision and Deep Learning techniques. The system is trained to classify ASL alphabet gestures from image data and aims to help bridge communication barriers between deaf or hard-of-hearing individuals and non-sign language users.

The project utilizes image classification methods on ASL hand sign datasets and applies Convolutional Neural Networks (CNN) for gesture recognition.

- Objectives

* Build a computer vision model capable of recognizing ASL alphabet gestures
* Train and evaluate a deep learning classification model
* Improve accessibility and communication using AI technology
* Explore image preprocessing and augmentation techniques for gesture recognition

- Dataset

This project uses the ASL Dataset from Kaggle:

[ASL Dataset on Kaggle](https://www.kaggle.com/datasets/ayuraj/asl-dataset/data?utm_source=chatgpt.com)

The dataset contains labeled images representing hand gestures for American Sign Language alphabets. It is designed for multi-class image classification tasks using computer vision approaches.

- Dataset Features

* Multiple ASL alphabet classes
* Image-based gesture data
* Suitable for CNN training
* Organized into training and testing folders

- Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Matplotlib
* Scikit-learn

- Project Workflow

1. Data Collection
The ASL image dataset was downloaded from Kaggle and organized into training and validation folders.

2. Data Preprocessing
Images were:
* Resized
* Normalized
* Augmented to improve generalization

3. Model Training
A Convolutional Neural Network (CNN) model was trained for multi-class classification of ASL hand gestures.

4. Evaluation
The model performance was evaluated using:
* Accuracy
* Loss curves
* Confusion matrix
* Validation metrics

- Model Architecture
The CNN model consists of:
* Convolutional layers
* Max pooling layers
* Fully connected dense layers
* Dropout regularization
* Softmax activation for classification

- Installation
Clone this repository:
```bash
git clone https://github.com/saaaileen/Computer-Vision-Project.git
cd Computer-Vision-Project
```
Install dependencies:
```bash
pip install -r requirements.txt
```
- How to Run
Train the Model
```bash
python train.py
```
Test / Predict
```bash
python predict.py
```

- Results

The model was able to classify ASL hand gestures with strong performance on validation data.

Potential improvements:
* Real-time webcam detection
* More advanced architectures such as ResNet or MobileNet
* Transfer learning
* Better augmentation techniques
* Deployment as a web or mobile application

- Project Structure

```bash
Computer-Vision-Project/
│
├── dataset/
├── models/
├── notebooks/
├── train.py
├── predict.py
├── requirements.txt
└── README.md
```

- Applications
This project can be applied to:
* Accessibility tools
* Educational platforms
* Real-time sign language translation systems
* Human-computer interaction systems

- References

* [ASL Dataset (Kaggle)](https://www.kaggle.com/datasets/ayuraj/asl-dataset/data?utm_source=chatgpt.com)
* TensorFlow Documentation
* OpenCV Documentation

- Contributors : Sisilia Annabel Aileen, Stella Budi Sugianto, Christina Angela Jodana, Alvin Lawrence, Nihal Singh, Nicholas Leonid, Vincent Christian Atmadjaya.

- Future Improvements
* Real-time webcam recognition
* Sentence-level sign recognition
* Mobile application deployment
* Transformer-based gesture recognition
* Improved dataset balancing and augmentation

