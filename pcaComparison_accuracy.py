import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data (high-dimensional)
n_samples = 1000
n_features = 10
np.random.seed(42)
X = np.random.randn(n_samples, n_features)

# Compute the first principal component using SVD for reference
U, S, Vt = np.linalg.svd(X, full_matrices=False)
true_pc = Vt[0]  # First principal component

# Normalize function
def normalize(v):
    return v / np.linalg.norm(v)

# Krasulina's Algorithm
def krasulina(X, lr=0.01, iterations=1000):
    w = np.random.randn(n_features)
    w = normalize(w)
    errors = []
    
    for t in range(iterations):
        i = np.random.randint(0, n_samples)
        x_t = X[i]
        y_t = np.dot(w, x_t)
        w += lr * (x_t - y_t * w)
        w = normalize(w)
        errors.append(np.linalg.norm(true_pc - np.dot(w, true_pc) * w))
    
    return w, errors

# Oja's Algorithm
def oja(X, lr=0.01, iterations=1000):
    w = np.random.randn(n_features)
    w = normalize(w)
    errors = []
    
    for t in range(iterations):
        i = np.random.randint(0, n_samples)
        x_t = X[i]
        y_t = np.dot(w, x_t)
        w += lr * y_t * (x_t - y_t * w)
        w = normalize(w)
        errors.append(np.linalg.norm(true_pc - np.dot(w, true_pc) * w))
    
    return w, errors

# Sanger's Algorithm
def sanger(X, lr=0.01, iterations=1000):
    W = np.random.randn(2, n_features)  # First 2 principal components
    W[0] = normalize(W[0])
    W[1] = normalize(W[1])
    errors = []
    
    for t in range(iterations):
        i = np.random.randint(0, n_samples)
        x_t = X[i]
        y = np.dot(W, x_t)
        W[0] += lr * y[0] * (x_t - y[0] * W[0])
        W[1] += lr * y[1] * (x_t - y[0] * W[0] - y[1] * W[1])
        W[0] = normalize(W[0])
        W[1] = normalize(W[1])
        errors.append(np.linalg.norm(true_pc - np.dot(W[0], true_pc) * W[0]))
    
    return W, errors

# Run all algorithms
iterations = 1000
_, err_krasulina = krasulina(X, iterations=iterations)
_, err_oja = oja(X, iterations=iterations)
_, err_sanger = sanger(X, iterations=iterations)

# Plot comparison
plt.figure(figsize=(10, 6))
plt.plot(err_krasulina, label="Krasulina's Algorithm")
plt.plot(err_oja, label="Oja's Algorithm")
plt.plot(err_sanger, label="Sanger's Algorithm")
plt.axhline(y=0, color='k', linestyle='--', label="SVD Reference")
plt.xlabel("Iterations")
plt.ylabel("Error from True PC")
plt.title("Convergence of PCA Algorithms")
plt.legend()
plt.show()