import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import curve_fit

# Generate synthetic data
def generate_data(n_samples, n_features):
    np.random.seed(42)
    return np.random.randn(n_samples, n_features)

# Time complexity models for curve fitting
def linear(x, a, b):
    return a * x + b

def quadratic(x, a, b, c):
    return a * x**2 + b * x + c

def cubic(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

def log_linear(x, a, b):
    return a * x * np.log(x) + b

# PCA Algorithms
def svd_pca(X):
    start_time = time.time()
    np.linalg.svd(X, full_matrices=False)
    return time.time() - start_time

def krasulina(X, iterations=1000, lr=0.01):
    start_time = time.time()
    n_features = X.shape[1]
    w = np.random.randn(n_features)
    w /= np.linalg.norm(w)
    
    for _ in range(iterations):
        x_t = X[np.random.randint(0, X.shape[0])]
        y_t = np.dot(w, x_t)
        w += lr * (x_t - y_t * w)
        w /= np.linalg.norm(w)

    return time.time() - start_time

def oja(X, iterations=1000, lr=0.01):
    start_time = time.time()
    n_features = X.shape[1]
    w = np.random.randn(n_features)
    w /= np.linalg.norm(w)
    
    for _ in range(iterations):
        x_t = X[np.random.randint(0, X.shape[0])]
        y_t = np.dot(w, x_t)
        w += lr * y_t * (x_t - y_t * w)
        w /= np.linalg.norm(w)

    return time.time() - start_time

def sanger(X, iterations=1000, lr=0.01):
    start_time = time.time()
    n_features = X.shape[1]
    W = np.random.randn(2, n_features)
    W[0] /= np.linalg.norm(W[0])
    W[1] /= np.linalg.norm(W[1])
    
    for _ in range(iterations):
        x_t = X[np.random.randint(0, X.shape[0])]
        y = np.dot(W, x_t)
        W[0] += lr * y[0] * (x_t - y[0] * W[0])
        W[1] += lr * y[1] * (x_t - y[0] * W[0] - y[1] * W[1])
        W[0] /= np.linalg.norm(W[0])
        W[1] /= np.linalg.norm(W[1])

    return time.time() - start_time

# Evaluate performance over increasing dataset sizes
sample_sizes = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]
times_svd = []
times_krasulina = []
times_oja = []
times_sanger = []

for n in sample_sizes:
    X = generate_data(n, 10)
    times_svd.append(svd_pca(X))
    times_krasulina.append(krasulina(X))
    times_oja.append(oja(X))
    times_sanger.append(sanger(X))

# Fit time complexity models
fit_svd, _ = curve_fit(quadratic, sample_sizes, times_svd)
fit_krasulina, _ = curve_fit(linear, sample_sizes, times_krasulina)
fit_oja, _ = curve_fit(linear, sample_sizes, times_oja)
fit_sanger, _ = curve_fit(quadratic, sample_sizes, times_sanger)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(sample_sizes, times_svd, 'o-', label="SVD (O(n²m))")
plt.plot(sample_sizes, times_krasulina, 's-', label="Krasulina (O(n))")
plt.plot(sample_sizes, times_oja, 'd-', label="Oja (O(n))")
plt.plot(sample_sizes, times_sanger, '^-', label="Sanger (O(n²))")

plt.xlabel("Number of Samples (n)")
plt.ylabel("Execution Time (seconds)")
plt.title("Empirical Time Complexity of PCA Algorithms")
plt.legend()
plt.grid()
plt.show()