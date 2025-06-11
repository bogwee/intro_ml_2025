# Machine Learning Project — Workflow

This document explains our day to day task, and what we have learned.

---

### 1. Day one

- Cloning the GitHub repository, setting up a virtual environment (venv) from scratch, and installing the necessary packages.

- Creating the Git environment

- Loading a scikit learn dataset of blurred digit images: load_digits(), the dataset contains the data (the pixel intensity decribing the number) and the labels (their numerical value because we are doing supervised learning)

- It is to be noted that the dataset is a range of blurred images of numbers (about 1797), the image itself is a 8x8 pixel display.

- When we extract the data (our X value) we are extracting a 2D array of 1797 elements (images) and 64 sub-elements (pixels), these are our images put into one array where the pixels are arranged in a orderly maner

- After extracting the X and Y matrix, we normalize the pixel values so that they range from [0, 16] (original range for black and white) to [0, 1].

- Our goal is to develop a machine learning model in Python by training it on the X data (features) and the corresponding Y labels (targets).

- Our features (X) is a (1797,64) array. To reduce noise, facilitate visualization, and train our model more effectively, we will apply Principal Component Analysis (PCA) to reduce the dimensionality of the data, meaning for each image array (64 values) we compute the covariance structure and thus extracting n componants describing the other 64-n pixels (which are projected onto these componants)reducing our features from (1797,64) to (1797,n).

- After transformation, the data is reduced to 2 dimensions for visualization only, if the reduced dimension is between 10 or 30 this enables us to apply machine learning methods more efficiently and with less complexity. When needed, we can use the inverse transform to restore the data to its original format, thus simplifying our overall workflow. 

- When reducing the dimensionality of the data, some error compared to the original matrix is inevitable. For this reason, it is important to select the most effective number of dimensions. To help with this, we can plot a graph of the cumulative explained variance versus the number of dimensions retained. This allows us to determine the optimal number of dimensions, in our case, the best choice is 28, keeping 95% of information at a reasonably low dimension.

- We can plot several graphs to compare our results: (1) a comparison between the original image of a digit and its reconstructed version after applying PCA and inverse transformation; (2) a graph of cumulative explained variance as a function of the number of dimensions; and (3) a scatter plot of our data projected onto two principal components, which shows how points with the same value tend to cluster together.