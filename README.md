# InfernoNet: Early Forest Fire Detection using Deep Learning

![Project Banner](https://github.com/user-attachments/assets/d235bd87-deaf-46ce-a8c2-bf05dfa15167)

**Project Grade: 86%**

InfernoNet is a deep learning project focused on the early detection of forest fires from drone footage. This repository contains the code, models, and documentation for a Convolutional Neural Network (CNN) trained to classify images as containing "Fire" or "No Fire".

## Table of Contents

- [About the Project](#about-the-project)
- [Dataset](#dataset)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## About the Project

The primary goal of InfernoNet was to develop a reliable and deployable neural network for integration into early warning systems for forest fires. By leveraging aerial imagery from drones, the system aims to provide rapid and accurate fire detection to help mitigate the devastating impact of wildfires.

For a comprehensive overview of the project, including the methodology, results, and conclusions, please refer to the [project report](Report.pdf).

## Dataset

The model was trained and tested using the [FLAME dataset](https://arxiv.org/pdf/2012.14036), which consists of video frames captured by a drone during prescribed pile burns in Northern Arizona, USA, in 2021.

### Data Preprocessing

The initial dataset required significant preprocessing:
- **Manual Relabeling:** 56% of the images initially labeled as "Lake" were found to contain forest scenery and were manually relabeled.
- **Data Restructuring:** The dataset was organized for both binary ("Fire" vs. "No Fire") and ternary ("Fire" vs. "Lake" vs. "No Fire") classification tasks.

<p align="center">
  <img src="https://github.com/user-attachments/assets/f342539f-ca93-4561-86dc-63f65d055ff4" width="400">
  <br>
  <em>Example of a mislabeled image that was corrected.</em>
</p>

## Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

Ensure you have Python 3.x installed. The required Python packages are listed in `requirements.txt`.

### Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/your_username/InfernoNet.git
    cd InfernoNet
    ```
2.  **Create a virtual environment (recommended):**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install the dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

## Project Structure

```
.
├── Binary/             # Notebooks for binary classification (Fire/No Fire)
├── Deployment/         # Files for model deployment
├── ModelMaking/        # Notebook for creating the CNN model
├── Simulation/         # Notebook for robustness simulations
├── Ternary/            # Notebooks for ternary classification (Fire/No Fire/Lake)
├── Weights/            # Trained model weights
├── .gitignore          # Files to be ignored by Git
├── LICENSE             # Project license
├── README.md           # This file
├── Report.pdf          # Detailed project report
└── requirements.txt    # Python dependencies
```

## Usage

The core of this project is in the Jupyter Notebooks. You can explore the different aspects of the project by running the notebooks in the respective directories.

-   **`ModelMaking/binary_model_making.ipynb`**: This notebook contains the code for building, training, and evaluating the CNN model.
-   **`Binary/*.ipynb`**: These notebooks are for experimenting with the binary classification task.
-   **`Ternary/ternary_rgb.ipynb`**: This notebook is for the ternary classification task.
-   **`Simulation/condition_simulations.ipynb`**: This notebook contains the robustness simulations.

To run a notebook, navigate to its directory and run:
```sh
jupyter notebook
```

## Model Architecture

A Convolutional Neural Network (CNN) was designed with the following architecture:

| Layer Type           | Output Shape       | Activation Function |
|----------------------|--------------------|---------------------|
| Input                | (254, 254, 3)      | -                   |
| Rescaling            | (254, 254, 3)      | -                   |
| Conv2D (32 filters)  | (254, 254, 32)     | LeakyReLU           |
| BatchNormalization   | (254, 254, 32)     | -                   |
| MaxPooling2D         | (127, 127, 32)     | -                   |
| Conv2D (64 filters)  | (127, 127, 64)     | LeakyReLU           |
| BatchNormalization   | (127, 127, 64)     | -                   |
| MaxPooling2D         | (63, 63, 64)       | -                   |
| Conv2D (64 filters)  | (63, 63, 64)       | LeakyReLU           |
| BatchNormalization   | (63, 63, 64)       | -                   |
| MaxPooling2D         | (31, 31, 64)       | -                   |
| Conv2D (32 filters)  | (31, 31, 32)       | LeakyReLU           |
| BatchNormalization   | (31, 31, 32)       | -                   |
| MaxPooling2D         | (15, 15, 32)       | -                   |
| Flatten              | (7200)             | -                   |
| Dense (32 units)     | (32)               | LeakyReLU           |
| Dropout (0.2)        | (32)               | -                   |
| Dense (1 unit)       | (1)                | Sigmoid             |

## Results

The model's performance and robustness are summarized in the following plots.

<p align="center">
  <img src="https://github.com/user-attachments/assets/9a6b80c7-d66a-45d6-94c7-607c3150d731" width="600">
  <br>
  <em>Model performance metrics.</em>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/fd49936a-189d-4323-a256-89b3a17d43bf" width="600">
  <br>
  <em>Robustness simulation results.</em>
</p>

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

-   The FLAME dataset authors.
-   Anyone else you would like to thank.