# Yoga-Pose-Classifier

# Yoga Pose Identification and Feedback System

## **Overview**
This project is a deep learning-based system designed to identify yoga poses from human body keypoints extracted from images or videos. The model not only classifies yoga poses but also provides feedback on alignment and accuracy, helping practitioners refine their techniques.

---

## **Features**
- **Pose Classification**: Identifies a variety of yoga poses with high accuracy.
- **Alignment Feedback**: Evaluates pose alignment and suggests corrections for improvement.
- **Real-Time Analysis (Future Goal)**: Integration for live feedback using webcam or video input.

---

## **Technologies Used**
- **Programming Language**: Python
- **Deep Learning Framework**: TensorFlow/Keras
- **Data Processing**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn
- **Pose Estimation**: Mediapipe or OpenPose for keypoint extraction

---

## **Model Architecture**
The deep neural network consists of the following:
- Input Layer: Accepts normalized keypoint data.
- Hidden Layers: Multiple dense layers with techniques such as Batch Normalization, Dropout, and Residual Connections.
- Output Layer: Softmax activation for multi-class classification.

**Optimizer**: Adam with a learning rate scheduler

**Loss Function**: Sparse categorical cross-entropy

---

## **Dataset**
### Data Description
- The dataset consists of keypoints extracted from images of yoga poses.
- Each sample includes 17 keypoints (x, y coordinates) and a corresponding pose label.

### Data Preprocessing
- Normalized coordinates relative to image dimensions.
- Augmented dataset with small rotations, flips, and scaling.
- Split into training, validation, and test sets.

---

## **Setup Instructions**
### Prerequisites
Ensure you have the following installed:
- Python 3.7 or higher
- TensorFlow 2.x
- Mediapipe (for keypoint extraction)
- NumPy, Pandas, Matplotlib




## **Results**
- **Accuracy**: Achieved 95.2% accuracy on the test set.
- **Performance Metrics**:
  | Metric    | Value |
  |-----------|-------|
  | Precision | 94.8% |
  | Recall    | 95.0% |
  | F1-Score  | 94.9% |

- **Visualization**:
  Loss and accuracy curves demonstrate smooth convergence.

---

## **Future Work**
1. **Real-Time Feedback**:
   - Develop a live analysis system using TensorFlow.js or a mobile application framework.
2. **Enhanced Dataset**:
   - Expand the dataset to include more yoga poses and practitioners.
3. **Transfer Learning**:
   - Incorporate pretrained models like EfficientNet or MobileNet.
4. **User Personalization**:
   - Adapt feedback based on user skill level and history.

---

## **Contributing**
Contributions are welcome! If you'd like to improve this project, follow these steps:
1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Submit a pull request.


---

---

## **Acknowledgments**
Special thanks to:
- TensorFlow and Keras teams for their robust frameworks.
- OpenPose and Mediapipe for providing efficient pose estimation solutions.

---

**Enjoy practicing and refining your yoga with AI!**

