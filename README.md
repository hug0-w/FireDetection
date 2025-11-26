# InfernoNet

The aim of this project was to develop a deployable neural network for use in early detection systems of forest fires.

**Project Grade: 86%**

## Dataset

This project uses the [FLAME dataset](https://arxiv.org/pdf/2012.14036), which contains frames from videos recorded by a drone in Northern Arizona, USA during prescribed pile burns.

The dataset is structured for two classification tasks:
1.  **Binary Classification**: "Fire" vs "No Fire"
2.  **Ternary Classification**: "Fire" vs "Lake" vs "No Fire"

Initial data required manual cleaning and relabeling, particularly for images containing lakes that were mislabeled as "No Fire".

## Project Structure

```
.
├── LICENSE
├── README.md
├── Report.pdf
├── requirements.txt
├── Binary/
│   ├── binary_grayscale.ipynb
│   ├── binary_red.ipynb
│   └── binary_rgb.ipynb
├── Deployment/
│   ├── binary_rgb.weights.h5
│   ├── Deployment.txt
│   └── models.py
├── ModelMaking/
│   └── binary_model_making.ipynb
├── Simulation/
│   └── condition_simulations.ipynb
├── Ternary/
│   └── ternary_rgb.ipynb
└── Weights/
    ├── binary_rgb.weights.h5
    └── ternary_weights.weights.h5
```

-   **Binary/**: Jupyter notebooks for binary classification experiments (grayscale, red channel, RGB).
-   **Ternary/**: Jupyter notebook for ternary classification (RGB).
-   **ModelMaking/**: Jupyter notebook detailing the process of building the CNN model. This notebook was developed in Google Colab.
-   **Deployment/**: Contains the model architecture (`models.py`) and saved weights for deployment.
-   **Simulation/**: Jupyter notebook for running robustness simulations.
-   **Weights/**: Saved model weights.
-   **Report.pdf**: The full project report.

## Getting Started

### Prerequisites

-   Python 3.x
-   pip

### Installation

1.  Clone the repository:
    ```sh
    git clone https://github.com/your-username/InfernoNet.git
    cd InfernoNet
    ```

2.  Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

The Jupyter notebooks in this repository are designed to be run in Google Colab.

-   **Model Training**: To train the models, open the `ModelMaking/binary_model_making.ipynb` notebook in Google Colab. This notebook contains the complete workflow, from data loading and preprocessing to model training and evaluation.
-   **Simulations**: The `Simulation/condition_simulations.ipynb` notebook can be used to run robustness simulations on the trained models.
-   **Model Architecture**: The `Deployment/models.py` file contains the Python code for the CNN model architectures. You can import the models from this file into your own scripts.

## Neural Network Architecture

A Convolutional Neural Network (CNN) was developed with the following design:

| Layer Type           | Output Shape       | Activation Function |
| -------------------- | ------------------ | --------------------- |
| Input                | (254, 254, 3)      | -                     |
| Rescaling            | (254, 254, 3)      | -                     |
| Conv2D (32 filters)  | (254, 254, 32)     | -                     |
| LeakyReLU            | (254, 254, 32)     | LeakyReLU             |
| BatchNormalization   | (254, 254, 32)     | -                     |
| MaxPooling2D         | (127, 127, 32)     | -                     |
| Conv2D (64 filters)  | (127, 127, 64)     | -                     |
| LeakyReLU            | (127, 127, 64)     | LeakyReLU             |
| BatchNormalization   | (127, 127, 64)     | -                     |
| MaxPooling2D         | (63, 63, 64)       | -                     |
| Conv2D (64 filters)  | (63, 63, 64)       | -                     |
| LeakyReLU            | (63, 63, 64)       | LeakyReLU             |
| BatchNormalization   | (63, 63, 64)       | -                     |
| MaxPooling2D         | (31, 31, 64)       | -                     |
| Conv2D (32 filters)  | (31, 31, 32)       | -                     |
| LeakyReLU            | (31, 31, 32)       | LeakyReLU             |
| BatchNormalization   | (31, 31, 32)       | -                     |
| MaxPooling2D         | (15, 15, 32)       | -                     |
| Flatten              | (7200)             | -                     |
| Dense (32 units)     | (32)               | -                     |
| LeakyReLU           | (32)               | LeakyReLU             |
| Dropout (0.2)        | (32)               | -                     |
| Dense (1 unit)       | (1)                | Sigmoid               |

## Results

For more details on the results, please look at the report (`Report.pdf`).

**Note to author: The images in the original README are broken. Please replace the placeholders below with the correct images.**

-   `[Image: Mislabeled data example]`
-   `[Image: Augmentation example]`
-   `[Image: Robustness simulation graph]`
-   `[Image: Results graph 1]`
-   `[Image: Results graph 2]`

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

-   This project was submitted for the PHAS0056 module at UCL.
-   The FLAME dataset is used for training and testing the models.