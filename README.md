# A Comparative Analysis of Classifiers for Medical Image Classification

### Group Members
- Srijan Kumar (2029202)
- Sneha Singh (2029203)
- Jyoti Kumari (2029204)

**Institution**: KIIT University | 2024

## Project Overview

Medical image classification is critical in healthcare to diagnose and treat diseases accurately. Recent advancements in deep learning have made transfer learning a powerful tool for medical image classification. This project performs a comparative analysis of several deep learning classifiers using transfer learning on two medical image datasets: one focused on skin diseases and the other on skin cancer. The goal is to determine the most effective classifier for medical image classification tasks.

## Datasets

- **Dataset 1**: Contains images of various skin diseases, categorized into 10 classes.
- **Dataset 2**: Comprises skin cancer images, categorized into 9 classes.

## Methodology

### Transfer Learning
Transfer learning involves using a pre-trained model that has learned features from a large dataset and adapting it to solve a different problem. In this project, we apply transfer learning to several models for medical image classification.

### Models Used
We evaluated the following pre-trained models:
1. VGG16
2. VGG19
3. InceptionV3
4. Inception ResNet V2
5. DenseNet121
6. DenseNet201

Each model's architecture was adapted using transfer learning techniques to handle the medical image classification tasks.

## Performance Metrics

We evaluated the models using the following metrics:

- **Accuracy**: Percentage of correctly classified images out of the total.
- **Precision**: Proportion of correctly classified positive samples among all samples classified as positive.
- **Recall**: Proportion of actual positive samples that were correctly classified.
- **F1 Score**: Harmonic mean of precision and recall, providing a balanced accuracy measure.
- **Kappa Score**: Measures agreement between predictions and actual classifications, accounting for chance agreement.
- **Confusion Matrix**: Summarizes the performance of the classification models by detailing true positives, true negatives, false positives, and false negatives.

## Results

- **DenseNet201** achieved the highest accuracy across both datasets, demonstrating its strong ability to classify medical images.
- **InceptionV3** and **DenseNet121** also performed well, with high accuracy and balanced precision, recall, and F1 scores.
  
### Summary of Results (Accuracy)
- **Dataset 1**: DenseNet201 achieved the highest accuracy.
- **Dataset 2**: DenseNet201 also outperformed other models in this dataset.

Graphs and confusion matrices were generated for a detailed analysis of the classifiers' performance.

## Future Scope

- Developing automated pipelines integrating transfer learning for medical image classification could reduce training time and resources, facilitating faster medical research and improved patient outcomes.
- Future research could explore the application of transfer learning to other medical imaging modalities such as MRI, CT scans, and ultrasound.
- Further improvements in model fine-tuning and dataset expansion could enhance the accuracy and efficiency of medical image analysis.

## Setup & Installation

### Prerequisites
- Python 3.x
- TensorFlow or PyTorch
- Jupyter Notebook (optional)

### Installation Steps
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/medical-image-classification.git
    cd medical-image-classification
    ```
2. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```
3. Download the datasets (Dataset 1 & Dataset 2) and place them in the appropriate directories.

4. Run the Jupyter notebooks or Python scripts to start the training and evaluation process:
    ```bash
    jupyter notebook analysis.ipynb
    ```

## Usage

1. Ensure that the datasets are properly loaded and pre-processed.
2. Select the desired model (VGG16, DenseNet121, etc.) for training and evaluation.
3. The system will automatically output the accuracy, precision, recall, F1 score, and confusion matrix for each model.
4. Visualizations of model performance, including accuracy graphs and confusion matrices, will be generated.

## References

1. R. Singh Chugh, V. Bhatia, K. Khanna, V. Bhatia, "A Comparative Analysis of Classifiers for Image Classification."
2. Z. Lai, H. Deng, "Medical Image Classification Based on Deep Features Extracted by Deep Model and Statistic Feature Fusion with Multilayer Perceptron."
3. M. Xin, Y. Wang, "Research on Image Classification Model Based on Deep Convolution Neural Network."

---
