import os
import time
import numpy as np
import cv2
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

# Constants
IMG_SIZE = (100, 100)

def load_images_with_identity(directory):
    images = []
    identities = []
    for identity in os.listdir(directory):
        identity_dir = os.path.join(directory, identity)
        if os.path.isdir(identity_dir):
            for filename in os.listdir(identity_dir):
                if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    continue
                img_path = os.path.join(identity_dir, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, IMG_SIZE)
                images.append(img.flatten())
                identities.append(identity)
    return np.array(images), identities

def apply_pca(data, n_components):
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(data)
    return pca, transformed

def authenticate_scan(scan_img, pca_model, transformed_data, identities):
    # Use the correct PCA transform logic
    scan_proj = transform_with_pca(scan_img, pca_model[0])  # pca_model[0] is the dict with mean and components
    dists = np.linalg.norm(transformed_data - scan_proj, axis=1)
    min_index = np.argmin(dists)
    return identities[min_index], dists[min_index]

def load_scan_images(directory):
    scans = []
    filenames = []
    for filename in os.listdir(directory):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            continue
        img_path = os.path.join(directory, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, IMG_SIZE)
        scans.append(img.flatten())
        filenames.append(filename)
    return scans, filenames

def svd_pca(data, n_components):
    """Perform PCA using Singular Value Decomposition (SVD)."""
    mean = np.mean(data, axis=0)
    centered_data = data - mean
    U, S, Vt = np.linalg.svd(centered_data, full_matrices=False)
    components = Vt[:n_components]
    transformed = np.dot(centered_data, components.T)
    return {'components': components, 'mean': mean}, transformed

def krasulina_pca(data, n_components, learning_rate=0.01, epochs=10):
    """Perform PCA using Krasulina's algorithm."""
    mean = np.mean(data, axis=0)
    centered_data = data - mean
    components = np.random.randn(n_components, centered_data.shape[1])
    for _ in range(epochs):
        for x in centered_data:
            components += learning_rate * np.outer(np.dot(x, components.T), (x - np.dot(x, components.T) @ components))
    transformed = np.dot(centered_data, components.T)
    return {'components': components, 'mean': mean}, transformed

def oja_pca(data, n_components, learning_rate=0.01, epochs=10):
    """Perform PCA using Oja's algorithm."""
    mean = np.mean(data, axis=0)
    centered_data = data - mean
    components = np.random.randn(n_components, centered_data.shape[1])
    for _ in range(epochs):
        for x in centered_data:
            delta = learning_rate * (np.dot(x, components.T).reshape(-1, 1) * x - np.diag(np.dot(components, components.T)).reshape(-1, 1) * components)
            components += delta
    transformed = np.dot(centered_data, components.T)
    return {'components': components, 'mean': mean}, transformed

def sanger_pca(data, n_components, learning_rate=0.01, epochs=10):
    """Perform PCA using Sanger's rule."""
    mean = np.mean(data, axis=0)
    centered_data = data - mean
    components = np.random.randn(n_components, centered_data.shape[1])
    for _ in range(epochs):
        for x in centered_data:
            y = np.dot(components, x)
            delta = np.zeros_like(components)
            for i in range(n_components):
                delta[i] += learning_rate * y[i] * (x - np.dot(components[:i+1].T, y[:i+1]))
            components += delta
    transformed = np.dot(centered_data, components.T)
    return {'components': components, 'mean': mean}, transformed

def transform_with_pca(scan_img, pca_model):
    """Transform a scan image using the PCA model."""
    centered_img = scan_img - pca_model['mean']
    return np.dot(centered_img, pca_model['components'].T)

if __name__ == "__main__":
    dataset_dir = './dataset'
    scan_dir = './scans'
    output_file = 'output.txt'

    dataset, dataset_labels = load_images_with_identity(dataset_dir)

    if len(dataset) == 0:
        raise ValueError("No images found in the dataset.")

    components = min(20, dataset.shape[1])

    pca_methods = {
        'SVD': lambda data: svd_pca(data, components),
        'Krasulina': lambda data: krasulina_pca(data, components),
        'Oja': lambda data: oja_pca(data, components),
        'Sanger': lambda data: sanger_pca(data, components),
    }

    scans, scan_filenames = load_scan_images(scan_dir)

    with open(output_file, 'w') as f:
        for method_name, pca_fn in pca_methods.items():
            start_train = time.time()
            pca_model, transformed_data = pca_fn(dataset)  # Returns the model and transformed data
            train_time = time.time() - start_train

            correct = 0
            total = 0
            start_test = time.time()
            for scan, fname in zip(scans, scan_filenames):
                identity, _ = authenticate_scan(scan, pca_model, transformed_data, dataset_labels)
                expected_identity = os.path.splitext(fname)[0].replace('_', ' ')
                if expected_identity.lower() in identity.lower():
                    correct += 1
                total += 1
            test_time = time.time() - start_test

            accuracy = correct / total if total else 0

            f.write(f"Method: {method_name}\n")
            f.write(f"Training Time: {train_time:.4f} seconds\n")
            f.write(f"Authentication Time: {test_time:.4f} seconds\n")
            f.write(f"Accuracy: {accuracy:.4f}\n\n")
