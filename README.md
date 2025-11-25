# DeepLearning-Classification-TransferLearning

![Python](https://img.shields.io/badge/python-3.8%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-EE4C2C?logo=pytorch&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-1.21%2B-013243?logo=numpy&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)

## Description
This repository contains the solutions for the third assignment of the Deep Learning course (Dr. Fatemizadeh, Sharif University of Technology). The project explores various deep learning architectures and techniques for image classification tasks.

It covers building simple Neural Networks (MLP) and Convolutional Neural Networks (CNN) from scratch to classify a custom dataset of shoes, sandals, and boots. Furthermore, it delves into advanced topics like **Transfer Learning** using pre-trained ResNet models on the CIFAR-10 dataset and explores **Knowledge Distillation** strategies.

## Features
* **Custom Image Classification (Part 1):**
    * Preprocessing and loading a "Shoe vs Sandal vs Boot" dataset (15k images).
    * Implementation of a Multi-Layer Perceptron (MLP) with dropout and ReLU activation.
    * Implementation of a custom Convolutional Neural Network (CNN).
    * Comparison of model performance (Accuracy, Precision, Recall).
* **Transfer Learning & Fine-Tuning (Part 2):**
    * Utilization of **ResNet50** pre-trained on ImageNet.
    * Fine-tuning strategies: Freezing feature extraction layers and training only the fully connected head vs. training the full model.
    * Application on the **CIFAR-10** dataset.
* **Advanced Techniques:**
    * **Knowledge Distillation (KD):** Experiments involving Teacher-Student network architectures (e.g., ResNet50 as teacher, ResNet18 as student).
    * **Visualization:** TensorBoard integration for tracking training loss and accuracy.

## Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/your-username/DeepLearning-Classification-TransferLearning.git](https://github.com/your-username/DeepLearning-Classification-TransferLearning.git)
    cd DeepLearning-Classification-TransferLearning
    ```

2.  Install the required dependencies:
    ```bash
    pip install torch torchvision numpy matplotlib tqdm scikit-learn
    ```

3.  (Optional) For TensorBoard visualization:
    ```bash
    pip install tensorboard
    ```

## Usage

### Part 1: Custom Dataset Classification
Open and run the Jupyter Notebook `DL2022_HW3_P1.ipynb`.
* **Data:** This notebook expects the "Shoe vs Sandal vs Boot Dataset". Ensure the data is placed in the correct directory or update the path in the `Prepairing` section.
* **Flags:** You can toggle specific parts of the training using the boolean flags defined at the start (e.g., `MLP = True`, `CNN = True`).

### Part 2: Transfer Learning on CIFAR-10
Open and run `DL2022_HW3_P2.ipynb`.
* **Data:** The notebook automatically downloads the CIFAR-10 dataset to the `./data` directory.
* **Training:** It demonstrates loading pre-trained weights (ResNet50) and fine-tuning them for the 10-class classification task.

### Theoretical Questions
The `DL2022_HW3_Teories.ipynb` file contains answers and derivations for the theoretical questions posed in the assignment.

## Contributing
Contributions are welcome!
1.  Fork the repository.
2.  Create a feature branch (`git checkout -b feature/NewArchitecture`).
3.  Commit your changes (`git commit -m 'Add DenseNet implementation'`).
4.  Push to the branch (`git push origin feature/NewArchitecture`).
5.  Open a Pull Request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
Created by [Nima Kelidari](https://github.com/nikelroid) - Deep Learning Course Project (EE Dept, Sharif University).
