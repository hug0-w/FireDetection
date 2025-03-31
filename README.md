# InfernoNet

The aim of this project was to develop a deployable neural network for use in 
early detection systems of forest fires.


## Dataset description:

The [FLAME dataset](https://arxiv.org/pdf/2012.14036) was used for training and testing , which contains frames from videos
recorded by a drone in Northern Arizona, USA 2021. The footage was recorded  during
prescribed pile burns, controlled fires used to create barriers to naturally spreading wild fires.

For the binary "Fire" vs "No_Fire" classification, the dataset is structured as follows:

<pre>FireNoFire                   
├── Training
│   ├── Fire
│   ├── No_Fire
└── Test
    ├── Fire
    ├── No_Fire </pre>

To perform a ternary classifcation, "Fire" vs "Lake" vs "No_Fire", the dataset was restructued as follows:
<pre>FireNoFire
├── Training
│   ├── Fire 
│   ├── Lake
│   ├── No_Fire
└── Test
    ├── Fire 
    ├── Lake
    ├── No_Fire </pre>

Initially, lake images were only found in Training/No_Fire/ and needed to be manually redistributed between Training/ and Test/ for both classification tasks.
56% of the data labelled as "Lake" actually contained forest scenery, requiring manul relabelling.

e.g  
<img src="https://github.com/user-attachments/assets/f342539f-ca93-4561-86dc-63f65d055ff4" width="400">


## Neural Network Architecture

A Convolutional Neural Network (CNN) was developed in ModelMaking/model_making.ipynb
with the following design:

<div align="center">

| Layer Type           | Output Shape       | Activation Function |
|----------------------|-------------------|---------------------|
| Input               | (254, 254, 3)      | -                   |
| Rescaling           | (254, 254, 3)      | -                   |
| Conv2D (32 filters) | (254, 254, 32)     | -                   |
| LeakyReLU           | (254, 254, 32)     | LeakyReLU           |
| BatchNormalization  | (254, 254, 32)     | -                   |
| MaxPooling2D        | (127, 127, 32)     | -                   |
| Conv2D (64 filters) | (127, 127, 64)     | -                   |
| LeakyReLU           | (127, 127, 64)     | LeakyReLU           |
| BatchNormalization  | (127, 127, 64)     | -                   |
| MaxPooling2D        | (63, 63, 64)       | -                   |
| Conv2D (64 filters) | (63, 63, 64)       | -                   |
| LeakyReLU           | (63, 63, 64)       | LeakyReLU           |
| BatchNormalization  | (63, 63, 64)       | -                   |
| MaxPooling2D        | (31, 31, 64)       | -                   |
| Conv2D (32 filters) | (31, 31, 32)       | -                   |
| LeakyReLU           | (31, 31, 32)       | LeakyReLU           |
| BatchNormalization  | (31, 31, 32)       | -                   |
| MaxPooling2D        | (15, 15, 32)       | -                   |
| Flatten             | (7200)             | -                   |
| Dense (32 units)    | (32)               | -                   |
| LeakyReLU          | (32)               | LeakyReLU           |
| Dropout (0.2)      | (32)               | -                   |
| Dense (1 unit)     | (1)                | Sigmoid             |

</div>

## Augmentation

<img src="https://github.com/user-attachments/assets/d235bd87-deaf-46ce-a8c2-bf05dfa15167" width="400">


## Robustness Simulations

<img src="https://github.com/user-attachments/assets/fa149902-24c5-481a-b4ea-7cc4f5c8132d" width="400">


