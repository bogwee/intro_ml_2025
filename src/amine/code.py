import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

##########################################
## Data loading and first visualisation
##########################################

# Load the handwritten digits dataset
digits = load_digits()

# Visualize some images
def first4():
    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.imshow(digits.images[i],cmap='gray')
    plt.show()

#first4()


# Display at least one random sample par class (some repetitions of class... oh well)
def plot_multi(data, y):
    '''Plots 16 digits'''
    nplots = 16
    nb_classes = len(np.unique(y))
    cur_class = 0
    fig = plt.figure(figsize=(15,15))
    for j in range(nplots):
        plt.subplot(4,4,j+1)
        to_display_idx = np.random.choice(np.where(y == cur_class)[0])
        plt.imshow(data[to_display_idx].reshape((8,8)), cmap='binary')
        plt.title(cur_class)
        plt.axis('off')
        cur_class = (cur_class + 1) % nb_classes
    plt.show()


#plot_multi(digits.data, digits.target)

##########################################
## Data exploration and first analysis
##########################################

def get_statistics_text(digits):
    print(f"Colonnes du dataset : {digits.keys()}")
    print(f"Targets possibles   : {digits.target_names}")
    print(f"Première image :\n{digits.images[0]}")
    print(f"Nombre total d'images : {digits.images.shape[0]}")

    unique, counts = np.unique(digits.target, return_counts=True)
    print("Nombre d'occurrences pour chaque chiffre :")
    for val, count in zip(unique, counts):
        print(f"{val} : {count} images")

#get_statistics_text(digits)


##########################################
## Start data preprocessing
##########################################

# Access the whole dataset as a matrix where each row is an individual (an image in our case) 
# and each column is a feature (a pixel intensity in our case)
## X = [
#  [Pixel1, Pixel2, ..., Pixel64],  # Image 1 as a row
#  [Pixel1, Pixel2, ..., Pixel64],  # Image 2 as a row
#  [Pixel1, Pixel2, ..., Pixel64],  # Image 3 as a row
#  [Pixel1, Pixel2, ..., Pixel64]   # Image 4 as a row
#]

# TODO: Create a feature matrix and a vector of labels
#label valeur exacte dans l'échantillon != taget valeur à predire
X = digits.data
y = digits.target

# Print dataset shape
print(f"Feature matrix shape: {X.shape}. Max value = {np.max(X)}, Min value = {np.min(X)}, Mean value = {np.mean(X)}")
print(f"Labels shape: {y.shape}")


# TODO: Normalize pixel values to range [0,1]
F = X/16  # Feature matrix after scaling bc its between 0 and 16

# Print matrix shape
print(f"Feature matrix F shape: {F.shape}. Max value = {np.max(F)}, Min value = {np.min(F)}, Mean value = {np.mean(F)}")

##########################################
##        Dimensionality reduction      ##
##########################################


### just an example to test, for various number of PCs
sample_index = 0
original_image = F[sample_index].reshape(8, 8)  # Reshape back to 8×8 for visualization

# TODO: Using the specific sample above, iterate the following:
# * Generate a PCA model with a certain value of principal components
# * Compute the approximation of the sample with this PCA model
# * Reconstruct a 64 dimensional vector from the reduced dimensional PCA space
# * Reshape the resulting approximation as an 8x8 matrix
# * Quantify the error in the approximation
# Finally: plot the original image and the 15 approximation on a 4x4 subfigure

pca = PCA(n_components=2)
F_pca = pca.fit_transform(F) 

#plot
plt.figure(figsize=(8,6))
scatter = plt.scatter(F_pca[:, 0], F_pca[:, 1], c=digits.target, cmap='tab10', alpha=0.7)
plt.legend(*scatter.legend_elements(), title="Chiffres")
plt.title("Projection PCA à 2 dimensions du dataset digits")
plt.xlabel("Composante principale 1")
plt.ylabel("Composante principale 2")
plt.grid(True)
plt.show()

F_ipca = pca.inverse_transform(F_pca)
print(f"L'erreur quadratique moyenne est :", mean_squared_error(F, F_ipca))

plt.subplot(1,2,1)
img0 = X[0].reshape(8,8)
plt.imshow(img0,cmap='gray')
plt.title("Image originale")

plt.subplot(1,2,2)
img1 = F_ipca[0].reshape(8,8)
plt.imshow(img1,cmap='gray')
plt.title("Image reconstruite")

plt.show()

##########################################
## Feature engineering
##########################################
### # Function to extract zone-based features
###  Zone-Based Partitioning is a feature extraction method
### that helps break down an image into smaller meaningful regions to analyze specific patterns.
def extract_zone_features(X):
    n = X.shape[0]
    res = np.zeros((n, 3))
    for i in range(n):
        zone1 = np.mean(X[i, 0:24])
        zone2 = np.mean(X[i, 24:40])
        zone3 = np.mean(X[i, 40:64])
        res[i, :] = [zone1, zone2, zone3]
    return res

# Apply zone-based feature extraction
F_zones = extract_zone_features(X)

# Print extracted feature shape
print(f"Feature matrix F_zones shape: {F_zones.shape}")


### Edge detection features
from skimage.filters import sobel
## TODO: Get used to the Sobel filter by applying it to an image and displaying both the original image 
# and the result of applying the Sobel filter side by side


# TODO: Compute the average edge intensity for each image and return it as an n by 1 array
def apply_sobel(X):
    n = X.shape[0]
    sobel_mean = np.zeros((n,1))
    for i in range(n):
        img = X[i].reshape(8, 8)
        sobel_img = sobel(img)
        sobel_mean[i] = np.mean(sobel_img)
    return sobel_mean
F_edges = apply_sobel(X)

# Print feature shape after edge extraction
print(f"Feature matrix F_edges shape: {F_edges.shape}")

### connect all the features together

# TODO: Concatenate PCA, zone-based, and edge features
def all_features(X_normalized):
    pca = PCA(n_components=20)
    X_pca = pca.fit_transform(X_normalized)
    X_zone = extract_zone_features(X_normalized)
    X_sobel = apply_sobel(X_normalized)
    X_all = np.hstack((X_pca, X_zone, X_sobel))
    return X_all

F_final = all_features(F)

print(f"Dimensions avec toutes les features :",F_final.shape)