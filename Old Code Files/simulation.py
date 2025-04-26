import os
import time
import numpy as np
import cv2
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import threading

# Constants
IMG_SIZE = (100, 100)


def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, IMG_SIZE)
    return img.flatten()


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
                img = preprocess_image(img_path)
                if img is not None:
                    images.append(img)
                    identities.append(identity)
    return np.array(images), identities


def authenticate_scan(scan_img, components, transformed_data, identities):
    scan_proj = scan_img @ components.T
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
        img = preprocess_image(img_path)
        if img is not None:
            scans.append(img)
            filenames.append(filename)
    return scans, filenames


# Online PCA implementations

def pca_svd(data, n_components):
    pca = PCA(n_components=n_components, svd_solver='full')
    pca.fit(data)
    return pca.components_, pca.transform(data)


def pca_krasulina(data, n_components, epochs=1, lr=1e-3):
    n_samples, n_features = data.shape
    components = np.random.randn(n_components, n_features)
    for epoch in range(epochs):
        print(f"Krasulina: Starting epoch {epoch + 1}/{epochs}")  # Debugging
        for idx, x in enumerate(data):
            for i in range(n_components):
                comp = components[i]
                proj = np.dot(comp, x)
                comp += lr * (proj * x - proj**2 * comp)
                norm = np.linalg.norm(comp)
                if norm > 0:
                    components[i] = comp / norm
            if idx % 100 == 0:  # Log progress every 100 samples
                print(f"Krasulina: Processed {idx + 1}/{n_samples} samples")
    return components, data @ components.T


def pca_oja(data, n_components, epochs=1, lr=1e-3):
    n_samples, n_features = data.shape
    components = np.random.randn(n_components, n_features)
    for epoch in range(epochs):
        print(f"Oja: Starting epoch {epoch + 1}/{epochs}")  # Debugging
        for idx, x in enumerate(data):
            for i in range(n_components):
                comp = components[i]
                proj = np.dot(comp, x)
                comp += lr * proj * (x - proj * comp)
                norm = np.linalg.norm(comp)
                if norm > 0:
                    components[i] = comp / norm
            if idx % 100 == 0:  # Log progress every 100 samples
                print(f"Oja: Processed {idx + 1}/{n_samples} samples")
    return components, data @ components.T


def pca_sanger(data, n_components, epochs=1, lr=1e-3):
    n_samples, n_features = data.shape
    components = np.random.randn(n_components, n_features)
    for epoch in range(epochs):
        print(f"Sanger: Starting epoch {epoch + 1}/{epochs}")  # Debugging
        for idx, x in enumerate(data):
            y = components @ x
            for i in range(n_components):
                xi = x - np.sum(y[:i+1, None] * components[:i+1], axis=0)
                components[i] += lr * y[i] * xi
                norm = np.linalg.norm(components[i])
                if norm > 0:
                    components[i] = components[i] / norm
            if idx % 100 == 0:  # Log progress every 100 samples
                print(f"Sanger: Processed {idx + 1}/{n_samples} samples")
    return components, data @ components.T


# Timeout handler
class TimeoutException(Exception):
    pass

def run_with_timeout(func, args=(), kwargs=None, timeout=60):
    """
    Runs a function with a timeout. If the function does not complete within the timeout,
    a TimeoutException is raised.
    """
    if kwargs is None:
        kwargs = {}

    result = [None]
    exception = [None]

    def target():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            exception[0] = e

    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        raise TimeoutException("PCA training exceeded time limit.")
    if exception[0]:
        raise exception[0]

    return result[0]


if __name__ == "__main__":
    dataset_dir = './dataset'
    scan_dir = './scans'
    output_file = 'output.txt'

    dataset, dataset_labels = load_images_with_identity(dataset_dir)

    if len(dataset) == 0:
        raise ValueError("No images found in the dataset.")

    components_num = min(95, dataset.shape[1])

    pca_methods = {
        'SVD': lambda data: pca_svd(data, components_num),
        'Krasulina': lambda data: pca_krasulina(data, components_num, epochs=5, lr=1e-3),
        'Oja': lambda data: pca_oja(data, components_num, epochs=5, lr=1e-3),
        'Sanger': lambda data: pca_sanger(data, components_num, epochs=5, lr=1e-3),
    }

    scans, scan_filenames = load_scan_images(scan_dir)

    with open(output_file, 'w') as f:
        for method_name, pca_fn in pca_methods.items():
            try:
                start_train = time.time()
                components, transformed_data = run_with_timeout(pca_fn, args=(dataset,), timeout=60)
                train_time = time.time() - start_train
            except TimeoutException as e:
                print(f"Error: {e}")
                f.write(f"Method: {method_name}\n")
                f.write("Training Time: Timeout\n")
                f.write("Authentication Time: N/A\n")
                f.write("Accuracy: N/A\n\n")
                continue
            except Exception as e:
                print(f"Unexpected error during {method_name} training: {e}")
                f.write(f"Method: {method_name}\n")
                f.write("Training Time: Error\n")
                f.write("Authentication Time: N/A\n")
                f.write("Accuracy: N/A\n\n")
                continue

            correct = 0
            total = 0
            start_test = time.time()
            for scan, fname in zip(scans, scan_filenames):
                identity, _ = authenticate_scan(scan, components, transformed_data, dataset_labels)
                expected_identity = os.path.splitext(fname)[0].replace('_', ' ').lower().strip()
                predicted_identity = identity.replace('_', ' ').lower().strip()  # Normalize predicted identity

                # Debugging: Log expected and predicted identities
                print(f"Expected: {expected_identity}, Predicted: {predicted_identity}")

                if expected_identity == predicted_identity:
                    correct += 1
                total += 1
            test_time = time.time() - start_test

            accuracy = correct / total if total else 0

            f.write(f"Method: {method_name}\n")
            f.write(f"Training Time: {train_time:.4f} seconds\n")
            f.write(f"Authentication Time: {test_time:.4f} seconds\n")
            f.write(f"Accuracy: {accuracy:.4f}\n\n")
