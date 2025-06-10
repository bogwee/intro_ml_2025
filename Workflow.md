# Machine Learning Project — Workflow

This document explains our day to day task, and what we have learned.

---

### 1. Day one

- Cloning the GitHub repository, setting up a virtual environment (venv) from scratch, and installing the necessary packages.

- Creating the Git environment

- Loading a scikit learn dataset of blurred digit images: load_digits(), the dataset contains the data (the pixel intensity decribing the number) and the labels (their numerical value because we are doing supervised learning)

- After loading the dataset, we normalize the pixel values so that they range from [0, 16] (original range for black and white) to [0, 1]. Once normalized, we extract a matrix X, where each row represents the pixel values of a specific digit image, and a vector Y containing the corresponding labels for each image. These constitute our features (X) and targets (Y) for model training.

- Our goal is to develop a machine learning model in Python by training it on the X data (features) and the corresponding Y labels (targets).

- Our dataset is a multi-dimensional matrix, with each instance described by 64 pixel intensity features. To reduce noise, facilitate visualization, and train our model more effectively, we will apply Principal Component Analysis (PCA) to reduce the dimensionality of the data.

- After transformation, the data is reduced to 2 dimensions, enabling us to apply machine learning methods more efficiently and with less complexity. When needed, we can use the inverse transform to restore the data to its original format, thus simplifying our overall workflow. 
