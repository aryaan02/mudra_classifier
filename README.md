# Bharatanatyam Mudra Classifier

## Introduction
Bharatanatyam is a classical dance form originating in Tamil Nadu, India. It is known for its grace, purity, tenderness, and sculpturesque poses, and the dance form is accompanied by Carnatic music.

This project aims to classify Bharatanatyam mudras (hand gestures) using Convolutional Neural Networks (CNNs). The dataset comes from the paper _Optimal feature selection and classification of Indian classical dance hand gesture dataset_ by R. Jisha Raj, Smitha Dharan, and T. T. Sunil, stored in a the GitHub repository [here](https://github.com/jisharajr/Bharatanatyam-Mudra-Dataset).

## Instructions
**To train the model:**
1. Clone this repository.
2. Clone the Bharatanatyam Mudra Dataset repository [here](https://github.com/jisharajr/Bharatanatyam-Mudra-Dataset), and place all files excluding the .md files in a 'data' directory in the root of this repository.
3. Create a virtual environment using `python -m venv env`.
4. Activate the virtual environment using `source env/bin/activate`.
5. Install the required packages using `pip install -r requirements.txt`.
6. Open the Jupyter notebook 'mudra_classification.ipynb' and run all cells.

**To use the camera to classify mudras:**
1. Follow all the steps above.
2. Run the Python script 'mudra__camera.py' using `python mudra_camera.py`.