import numpy as np
import matplotlib.pyplot as plt
import time

# Generate synthetic data
def generate_data(n_samples, n_features):
    np.random.seed(42)
    return np.random.randn(n_samples, n_features)

# Online PCA Error Tracking
def krasulina(X, iterations=1000, lr=0.01):
    n_features = X.shape[1]
    w = np.random.randn(n_features)
    w /= np.linalg.norm(w)

    errors = []
    for _ in range(iterations):
        x_t = X[np.random.randint(0, X.shape[0])]
        y_t = np.dot(w, x_t)
        w += lr * (x_t - y_t * w)
        w /= np.linalg.norm(w)
        errors.append(1 - np.abs(np.dot(w, np.linalg.svd(X, full_matrices=False)[2][0])))  # Error vs True PC

    return errors

def oja(X, iterations=1000, lr=0.01):
    n_features = X.shape[1]
    w = np.random.randn(n_features)
    w /= np.linalg.norm(w)

    errors = []
    for _ in range(iterations):
        x_t = X[np.random.randint(0, X.shape[0])]
        y_t = np.dot(w, x_t)
        w += lr * y_t * (x_t - y_t * w)
        w /= np.linalg.norm(w)
        errors.append(1 - np.abs(np.dot(w, np.linalg.svd(X, full_matrices=False)[2][0])))  # Error vs True PC

    return errors

def sanger(X, iterations=1000, lr=0.01):
    n_features = X.shape[1]
    W = np.random.randn(2, n_features)
    W[0] /= np.linalg.norm(W[0])
    W[1] /= np.linalg.norm(W[1])

    errors = []
    for _ in range(iterations):
        x_t = X[np.random.randint(0, X.shape[0])]
        y = np.dot(W, x_t)
        W[0] += lr * y[0] * (x_t - y[0] * W[0])
        W[1] += lr * y[1] * (x_t - y[0] * W[0] - y[1] * W[1])
        W[0] /= np.linalg.norm(W[0])
        W[1] /= np.linalg.norm(W[1])
        errors.append(1 - np.abs(np.dot(W[0], np.linalg.svd(X, full_matrices=False)[2][0])))  # Error vs True PC

    return errors

# Compare Convergence of Online PCA Algorithms
X = generate_data(5000, 10)
iterations = 1000
errors_krasulina = krasulina(X, iterations)
errors_oja = oja(X, iterations)
errors_sanger = sanger(X, iterations)

plt.figure(figsize=(10, 6))
plt.plot(errors_krasulina, label="Krasulina")
plt.plot(errors_oja, label="Oja")
plt.plot(errors_sanger, label="Sanger")
plt.xlabel("Iterations")
plt.ylabel("Error (1 - Cosine Similarity with True PC)")
plt.title("Convergence of Online PCA Algorithms")
plt.legend()
plt.grid()
plt.show()

# Compare Execution Time vs. Sample Size
sample_sizes = [100, 500, 1000, 5000, 10000, 20000, 100000, 1000000]
times_svd = []
times_krasulina = []
times_oja = []
times_sanger = []

for n in sample_sizes:
    X = generate_data(n, 10)
    
    start = time.time()
    np.linalg.svd(X, full_matrices=False)
    times_svd.append(time.time() - start)
    
    start = time.time()
    krasulina(X, iterations=1000)
    times_krasulina.append(time.time() - start)
    
    start = time.time()
    oja(X, iterations=1000)
    times_oja.append(time.time() - start)
    
    start = time.time()
    sanger(X, iterations=1000)
    times_sanger.append(time.time() - start)

plt.figure(figsize=(10, 6))
plt.plot(sample_sizes, times_svd, 'o-', label="SVD (Batch, O(n²m))")
plt.plot(sample_sizes, times_krasulina, 's-', label="Krasulina (Online, O(n))")
plt.plot(sample_sizes, times_oja, 'd-', label="Oja (Online, O(n))")
plt.plot(sample_sizes, times_sanger, '^-', label="Sanger (Online, O(n²))")

plt.xlabel("Number of Samples (n)")
plt.ylabel("Execution Time (seconds)")
plt.title("Execution Time vs. Sample Size for PCA Algorithms")
plt.legend()
plt.grid()
plt.show()