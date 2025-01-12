# Capstone_Project_Deep_Learning_for_Computer_Vision

# Project Title:
**DeepFER: Facial Emotion Recognition Using Deep Learning**

## Overview

DeepFER: Facial Emotion Recognition Using Deep Learning aims to develop a robust and efficient system for recognizing emotions from facial expressions using advanced deep learning techniques. This project leverages Convolutional Neural Networks (CNNs) and Transfer Learning to accurately classify emotions such as happiness, sadness, anger, surprise, and more from images of human faces. The system will be trained on a diverse dataset of facial images, employing data augmentation and fine-tuning methods to enhance its performance. By integrating state-of-the-art computer vision algorithms and neural network architectures, DeepFER seeks to achieve high accuracy and real-time processing capabilities. The ultimate goal is to create a versatile tool that can be applied in various fields, including human-computer interaction, mental health monitoring, and customer service, enhancing the way machines understand and respond to human emotions.

## Project Background

Customer satisfaction in the e-commerce sector is a pivotal metric that influences loyalty, repeat business, and word-of-mouth marketing. Traditionally, companies have relied on direct surveys to gauge customer satisfaction, which can be time-consuming and may not always capture the full spectrum of customer experiences. With the advent of deep learning, it's now possible to predict customer satisfaction scores in real-time, offering a granular view of service performance and identifying areas for immediate improvement.

In recent years, the field of facial emotion recognition has gained significant attention due to its wide range of applications in various domains, including mental health monitoring, human-computer interaction, customer service, and security. Emotion recognition from facial expressions is a challenging task, as it involves accurately identifying subtle differences in facial features corresponding to different emotional states. Traditional methods relied heavily on handcrafted features and rule-based approaches, which often lacked the ability to generalize across diverse datasets and real-world scenarios.

The advent of deep learning, particularly Convolutional Neural Networks (CNNs), has revolutionized the way facial emotion recognition systems are developed. CNNs have demonstrated exceptional performance in image classification tasks by automatically learning hierarchical feature representations from raw data.

This project, DeepFER: Facial Emotion Recognition Using Deep Learning, aims to harness the power of CNNs and Transfer Learning to build a robust and efficient facial emotion recognition system. By training the model on large, annotated datasets and employing advanced techniques such as data augmentation and fine-tuning, DeepFER aspires to achieve high accuracy and real-time processing capabilities.

The motivation behind this project stems from the growing need for automated systems that can understand and respond to human emotions effectively. Such systems can significantly enhance user experiences in various applications, from interactive virtual assistants to personalized mental health interventions. DeepFER seeks to bridge the gap between advanced AI techniques and practical emotion recognition applications, paving the way for more intuitive and empathetic machine interactions with humans.

## Dataset Overview

- **Dataset Composition:**
  - Contains images categorized into seven distinct emotion classes: angry, sad, happy, fear, neutral, disgust, and surprise.
- **Emotion Classes:**
  - **Angry:** Images depicting expressions of anger.
  - **Sad:** Images depicting expressions of sadness.
  - **Happy:** Images depicting expressions of happiness.
  - **Fear:** Images depicting expressions of fear.
  - **Neutral:** Images depicting neutral, non-expressive faces.
  - **Disgust:** Images depicting expressions of disgust.
  - **Surprise:** Images depicting expressions of surprise.

- **Image Characteristics:**
  - High-quality facial images with diverse backgrounds and lighting conditions.
  - Includes both posed and spontaneous expressions to ensure robustness.
- **Data Augmentation:**
  - Techniques such as rotation, scaling, and flipping applied to increase dataset variability and enhance model generalization.
- **Dataset Annotations:**
  - Each image is labeled with its corresponding emotion class.
- **Data Source:**
  - Collected from publicly available facial expression databases and crowd-sourced contributions.
- **Usage:**
  - Used for training, validation, and testing phases in the emotion recognition model development.
- **Purpose:**
  - To train and evaluate the DeepFER model for accurate and real-time facial emotion recognition across diverse scenarios.





## Project Goal

The primary goal of **DeepFER: Facial Emotion Recognition Using Deep Learning** is to develop an advanced and efficient system capable of accurately identifying and classifying human emotions from facial expressions in real-time. By leveraging state-of-the-art Convolutional Neural Networks (CNNs) and Transfer Learning techniques, this project aims to create a robust model that can handle the inherent variability in facial expressions and diverse image conditions. The system will be trained on a comprehensive dataset featuring seven distinct emotions: angry, sad, happy, fear, neutral, disgust, and surprise. The ultimate objective is to achieve high accuracy and reliability, making DeepFER suitable for applications in human-computer interaction, mental health monitoring, customer service, and beyond. Through this project, we aim to bridge the gap between cutting-edge AI research and practical emotion recognition applications, contributing to more empathetic and responsive machine interactions with humans.



#### Summary of Data:
- **Classes**: The dataset contains 7 classes: `['angry', 'surprise', 'happy', 'disgust', 'fear', 'neutral', 'sad']`.
- **Total Train Images**: 28,821 images.
- **Image Size**: All images are uniformly sized at **48x48 pixels**.
- **Image Mode**: Grayscale (`L`), meaning each image has a single channel (shape: `48x48x1`).
- **Class Distribution**:
  - The `happy` class has the highest representation, comprising approximately 25% of the dataset.
  - The `disgust` class has the lowest representation, contributing around 1.5% of the dataset.

This dataset is well-suited for tasks such as facial emotion recognition and is already preprocessed to a consistent size and format.



## Deployment
Deployment -->
The deployment process is carried out within the `app.ipynb` notebook.




## Conclusion:

The development and evaluation of four models for Facial Emotion Recognition (FER) provided valuable insights into leveraging deep learning techniques for emotion classification. Across all iterations, significant improvements were made in both model performance and generalization through various strategies, including architecture selection, data augmentation, and fine-tuning.

Key Observations:
1. Performance Progression:
   - Each successive model iteration incorporated improvements in architecture or training methodology.
   - The final model, based on ResNet34, achieved a validation accuracy of 61.1%, marking a considerable improvement over earlier models.
2. Impact of Transfer Learning:
   - Transfer learning significantly boosted performance by utilizing pre-trained features, reducing training time and improving convergence.
3. Challenges:
   - Consistent challenges included differentiating between visually similar emotions (e.g., sadness vs. neutral, fear vs. surprise).
   - Limited validation accuracy indicates potential issues with class imbalance, dataset quality, or the need for more advanced architectures.

Model Summaries:
- Model 1: A custom CNN baseline model achieving ~35% validation accuracy, providing a starting point for FER exploration.  
- Model 2: CNN-based model that improved accuracy (~50%) using transfer learning and a robust feature extractor.  
- Model 3: ResNet50V2 model with advanced augmentation and partial layer freezing, reaching ~55% accuracy.  
- Model 4: ResNet34 fine-tuned with optimized augmentations and hyperparameters, achieving the highest accuracy of 61.1%.

Future Work:

1. Dataset Enhancements
- Class Balance:
  - Analyze class distribution to identify and address imbalance using techniques like weighted loss functions or oversampling.
- Data Diversity:
  - Incorporate more diverse datasets that include varied lighting, angles, and cultural contexts for better generalization.
- Synthetic Data Generation:
  - Utilize Generative Adversarial Networks (GANs) to generate synthetic images for underrepresented classes.

2. Advanced Architectures
- Deeper CNN Models:
  - Experiment with deeper architectures such as DenseNet, or EfficientNet.
- Vision Transformers (ViTs):
  - Investigate the use of attention-based models for improved performance in learning subtle distinctions between emotions.
- Hybrid Models:
  - Combine CNNs with ViTs or Recurrent Neural Networks (RNNs) to capture both spatial and temporal patterns in emotions.

3. Hyperparameter Optimization
- Use advanced optimization techniques like Bayesian Optimization or Optuna to fine-tune learning rates, dropout rates, and other hyperparameters.

4. Regularization Techniques
- Implement additional regularization strategies such as Dropout, Label Smoothing, and Weight Decay to reduce overfitting risks.

5. Ensemble Learning
- Develop an ensemble of models, combining the strengths of multiple architectures to improve robustness and accuracy.

6. Explainability and Interpretability
- Use explainability tools like Grad-CAM or SHAP to visualize which features the model uses to classify emotions, enhancing trust in real-world applications.

7. Real-World Deployment
- Deploy the model in real-world scenarios (e.g., healthcare, customer service) to evaluate its robustness in dynamic environments.
- Optimize for edge devices using techniques like model quantization or pruning for deployment efficiency.





The four models demonstrated progressive advancements in recognizing human emotions, achieving promising results with a validation accuracy of up to 61.1%. While this accuracy highlights the potential of deep learning in FER tasks, there is room for improvement to achieve state-of-the-art performance. By addressing the outlined challenges and leveraging advancements in computer vision, future iterations can create a highly robust and accurate FER system suitable for diverse real-world applications.

