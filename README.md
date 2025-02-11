# Breast-Cancer-Classification-using-CNN
Classifying tumours in breasts and predicting if the tumour is benign or malignant  

## Overview
Breast cancer is one of the leading causes of cancer-related deaths in women worldwide. Early and accurate detection plays a crucial role in improving survival rates. This project leverages **Convolutional Neural Networks (CNNs)** to classify breast cancer images as **benign or malignant**, assisting radiologists in making more informed decisions.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Future Enhancements](#future-enhancements)
- [Contributors](#contributors)
- [License](#license)

## Features
✔ Automated classification of breast cancer images (benign vs malignant).  
✔ Uses CNN for feature extraction and classification.  
✔ Image preprocessing techniques for noise reduction and enhancement.  
✔ High accuracy achieved using deep learning techniques.  
✔ Helps reduce human error in breast cancer diagnosis.  

## Dataset
The dataset consists of **mammographic images** categorized as benign or malignant. Preprocessing techniques are applied to improve image quality before feeding them into the CNN model.

## Technologies Used
- **Programming Language**: Python
- **Frameworks & Libraries**:
  - TensorFlow/Keras (for CNN model training)
  - OpenCV (for image preprocessing)
  - NumPy, Pandas (for data manipulation)
  - Matplotlib, Seaborn (for data visualization)
- **Jupyter Notebook / Spyder IDE**

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/hrithiksai18/Breast-Cancer-Classification-using-CNN.git
   cd Breast-Cancer-Classification-using-CNN
   ```
2. Install required dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook or Python script to train the model.

## Usage
1. Load the dataset and preprocess the images.
2. Train the CNN model using the preprocessed data.
3. Evaluate the model's performance on the test dataset.
4. Use the trained model for predicting new mammographic images.

## Results
The CNN model achieves high accuracy in distinguishing between benign and malignant tissues, reducing false positives and unnecessary biopsies. The evaluation metrics include:
- Accuracy: **~97%**
- Precision, Recall, and F1-Score for performance measurement.

## Future Enhancements
🔹 Improve the model by experimenting with advanced architectures (e.g., ResNet, EfficientNet).  
🔹 Implement real-time detection with a web or mobile interface.  
🔹 Expand the dataset for better generalization.  

## Contributors
- **Hrithik Sai Grandhisiri** ([GitHub](https://github.com/hrithiksai18))
- **Team Members**

## License
This project is open-source and available under the **MIT License**.
