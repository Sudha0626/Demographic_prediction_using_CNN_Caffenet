# 🎯 Deep Learning Pipeline for Demographic Inference (CNN)

A computer vision pipeline using Convolutional Neural Networks (CNN) and OpenCV for real-time facial detection and demographic inference — designed for applications in smart surveillance and personalized user experiences.

---

## 📌 Overview

This project implements a multi-model deep learning pipeline that performs three sequential tasks in a single pass: detect faces in an image, estimate the age range, and classify the gender. The pipeline uses pre-trained Caffe models for age and gender inference on top of OpenCV's face detector.

**Key capabilities:**
- Real-time facial detection using OpenCV DNN
- Age estimation across multiple age-range categories
- Gender classification (Male / Female)
- Multi-face support — processes all detected faces in a single image
- Works on static images

---

## 🛠 Tech Stack

| Component | Technology |
|---|---|
| Face Detection | OpenCV DNN (opencv_face_detector) |
| Age Estimation | Pre-trained Caffe Model (age_net) |
| Gender Classification | Pre-trained Caffe Model (gender_net) |
| Language | Python |

---

## 🏗 Pipeline Architecture

```
Input Image
      ↓
OpenCV Face Detector (opencv_face_detector_uint8.pb)
      ↓
Face ROI Extraction (per detected face)
      ↓
  ┌───────────────────┬───────────────────┐
  ↓                   ↓
Age Net           Gender Net
(Caffe)           (Caffe)
  ↓                   ↓
Age Range         Gender Label
  └───────────────────┘
            ↓
  Annotated Output Image
```

---

## ⚙️ How to Run Locally

```bash
# Clone the repo
git clone https://github.com/Sudha0626/Face-detection-with-gender-and-age.git
cd Face-detection-with-gender-and-age

# Install dependencies
pip install opencv-python

# Run detection on an image
python detect.py --image your_image.jpg
```

---

## 📁 Project Structure

```
Face-detection-with-gender-and-age/
├── detect.py                        # Main inference script
├── opencv_face_detector_uint8.pb    # Face detection model
├── opencv_face_detector.pbtxt       # Face detector config
├── age_net.caffemodel               # Age estimation model
├── age_deploy.prototxt              # Age model config
├── gender_net.caffemodel            # Gender classification model
├── gender_deploy.prototxt           # Gender model config
├── combined.jpg                     # Sample output image
├── [test images]                    # Sample input images
└── README.md
```

---

## 📊 Age Categories

The model classifies age into the following ranges:
`(0-2)` · `(4-6)` · `(8-12)` · `(15-20)` · `(25-32)` · `(38-43)` · `(48-53)` · `(60-100)`

---

## 🌱 Future Improvements

- Real-time webcam inference
- Fine-tune models on a more diverse dataset for improved accuracy
- Add emotion detection as an additional inference head
- Package as a deployable Streamlit or FastAPI application

---

## 👩‍💻 Author

**Raaga Sudha** — [LinkedIn](https://linkedin.com/in/raagasudha06) · [GitHub](https://github.com/Sudha0626)
 
