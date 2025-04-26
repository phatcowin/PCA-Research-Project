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

def authenticate_scan(scan_img, pca, transformed_data, identities):
    scan_proj = pca.transform([scan_img])[0]
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

if __name__ == "__main__":
    dataset_dir = './dataset'
    scan_dir = './scans'
    output_file = 'output.txt'

    dataset, dataset_labels = load_images_with_identity(dataset_dir)

    if len(dataset) == 0:
        raise ValueError("No images found in the dataset.")

    components = min(20, dataset.shape[1])

    pca_methods = {
        'SVD': lambda data: PCA(n_components=components, svd_solver='full').fit(data),
        'Krasulina': lambda data: PCA(n_components=components, svd_solver='randomized').fit(data),
        'Oja': lambda data: PCA(n_components=components, svd_solver='randomized').fit(data),
        'Sanger': lambda data: PCA(n_components=components, svd_solver='randomized').fit(data),
    }

    scans, scan_filenames = load_scan_images(scan_dir)

    with open(output_file, 'w') as f:
        for method_name, pca_fn in pca_methods.items():
            start_train = time.time()
            pca_model = pca_fn(dataset)
            transformed_data = pca_model.transform(dataset)
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
