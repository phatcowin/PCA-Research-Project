import numpy as np
import matplotlib.pyplot as plt
import time

# Generate synthetic data (high-dimensional)
n_samples = 1000
n_features = 10
np.random.seed(42)
X = np.random.randn(n_samples, n_features)

# Compute the first principal component using SVD for reference
def svd_pca(X):
    start_time = time.time()
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    execution_time = time.time() - start_time
    return Vt[0], execution_time  # First principal component

# Normalize function
def normalize(v):
    return v / np.linalg.norm(v)

# Krasulina's Algorithm
def krasulina(X, lr=0.01, iterations=1000):
    start_time = time.time()
    w = np.random.randn(n_features)
    w = normalize(w)
    
    for t in range(iterations):
        i = np.random.randint(0, n_samples)
        x_t = X[i]
        y_t = np.dot(w, x_t)
        w += lr * (x_t - y_t * w)
        w = normalize(w)
    
    execution_time = time.time() - start_time
    return w, execution_time

# Oja's Algorithm
def oja(X, lr=0.01, iterations=1000):
    start_time = time.time()
    w = np.random.randn(n_features)
    w = normalize(w)
    
    for t in range(iterations):
        i = np.random.randint(0, n_samples)
        x_t = X[i]
        y_t = np.dot(w, x_t)
        w += lr * y_t * (x_t - y_t * w)
        w = normalize(w)
    
    execution_time = time.time() - start_time
    return w, execution_time

# Sanger's Algorithm
def sanger(X, lr=0.01, iterations=1000):
    start_time = time.time()
    W = np.random.randn(2, n_features)  # First 2 principal components
    W[0] = normalize(W[0])
    W[1] = normalize(W[1])
    
    for t in range(iterations):
        i = np.random.randint(0, n_samples)
        x_t = X[i]
        y = np.dot(W, x_t)
        W[0] += lr * y[0] * (x_t - y[0] * W[0])
        W[1] += lr * y[1] * (x_t - y[0] * W[0] - y[1] * W[1])
        W[0] = normalize(W[0])
        W[1] = normalize(W[1])
    
    execution_time = time.time() - start_time
    return W, execution_time

# Run all algorithms
iterations = 1000
_, time_krasulina = krasulina(X, iterations=iterations)
_, time_oja = oja(X, iterations=iterations)
_, time_sanger = sanger(X, iterations=iterations)
_, time_svd = svd_pca(X)

# Plot comparison
algorithms = ["Krasulina's Algorithm", "Oja's Algorithm", "Sanger's Algorithm", "SVD"]
times = [time_krasulina, time_oja, time_sanger, time_svd]

plt.figure(figsize=(10, 6))
plt.bar(algorithms, times, color=['blue', 'green', 'red', 'purple'])
plt.xlabel("Algorithms")
plt.ylabel("Execution Time (seconds)")
plt.title("Time Complexity Comparison of PCA Algorithms")
plt.show()