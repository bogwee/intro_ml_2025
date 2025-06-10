# Machine Learning Project — Workflow

This document explains our day to day task, and what we have learned.

---

### 1. Day one

- Cloning the GitHub repository, setting up a virtual environment (venv) from scratch, and installing the necessary packages.

- Creating the Git environment

- Loading a scikit learn dataset of blurred digit images: load_digits(), the dataset contains the data (the pixel intensity decribing the number) and the labels (their numerical value because we are doing supervised learning)

- Once the dataset loaded we normalize the data to have values within [0,1] instead of [0,255], once normalized we extract a matrix containing the pixel values on each row of a specific number and a vector containing the label of each number these are our X and Y values.

- Our objective is going to be to create a Model by plugging in the X data and Y labels, training this model using machine learning skills in python.

