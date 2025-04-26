import numpy as np
import time
import matplotlib.pyplot as plt

# ------------------------------
# PCA Algorithms implementations
# ------------------------------
def pca_svd(X, n_components):
    """Perform PCA using SVD"""
    X_centered = X - np.mean(X, axis=0)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    components = Vt[:n_components]
    return components

def pca_krasulina(X, n_components, max_iter=1000, learning_rate=0.01):
    """Implement Krasulina's method for PCA"""
    n_samples, n_features = X.shape
    X_centered = X - np.mean(X, axis=0)
    
    components = np.random.rand(n_components, n_features)
    components, _ = np.linalg.qr(components.T)
    components = components.T
    
    for _ in range(max_iter):
        idx = np.random.randint(0, n_samples)
        x = X_centered[idx]
        
        for j in range(n_components):
            projection = np.dot(x, components[j])
            components[j] += learning_rate * (projection * x - projection**2 * components[j])
            
            if j > 0:
                for k in range(j):
                    components[j] -= np.dot(components[j], components[k]) * components[k]
            
            norm = np.linalg.norm(components[j])
            if norm > 0:
                components[j] /= norm
    
    return components

def pca_oja(X, n_components, max_iter=1000, learning_rate=0.01):
    """Implement Oja's method for PCA"""
    n_samples, n_features = X.shape
    X_centered = X - np.mean(X, axis=0)
    
    components = np.zeros((n_components, n_features))
    
    for j in range(n_components):
        w = np.random.rand(n_features)
        w /= np.linalg.norm(w)
        
        # Deflate the data for subsequent components
        if j > 0:
            X_proj = np.dot(X_centered, components[:j].T)
            X_approx = np.dot(X_proj, components[:j])
            X_residual = X_centered - X_approx
        else:
            X_residual = X_centered
            
        for _ in range(max_iter):
            idx = np.random.randint(0, n_samples)
            x = X_residual[idx]
            
            y = np.dot(w, x)
            w += learning_rate * y * (x - y * w)
            w /= np.linalg.norm(w)
        
        components[j] = w
    
    return components

def pca_sanger(X, n_components, max_iter=1000, learning_rate=0.01):
    """Implement Sanger's method (GHA) for PCA"""
    if len(X) == 0:
        raise ValueError("Empty training set")
        
    n_samples, n_features = X.shape
    X_centered = X - np.mean(X, axis=0)
    
    components = np.random.rand(n_components, n_features)
    components /= np.linalg.norm(components, axis=1, keepdims=True)
    
    for _ in range(max_iter):
        idx = np.random.randint(0, n_samples)
        x = X_centered[idx]
        
        y = np.dot(components, x)
        
        for j in range(n_components):
            # Calculate lower triangular y*w terms
            lower_sum = np.zeros(n_features)
            for k in range(j+1):
                lower_sum += y[k] * components[k]
            
            # Update rule
            components[j] += learning_rate * y[j] * (x - lower_sum)
            
            # Normalize
            components[j] /= np.linalg.norm(components[j])
    
    return components

def pca_sanger_optimized(X, n_components, max_iter=500, initial_lr=0.01, 
                         lr_decay=0.01, batch_size=16):
    """Time-constrained implementation of Sanger's method with safety measures"""
    n_samples, n_features = X.shape
    X_centered = X - np.mean(X, axis=0)
    
    # Scale data to prevent numerical issues
    scale = np.max(np.abs(X_centered)) + 1e-10
    X_centered = X_centered / scale
    
    # Use SVD for initialization - this gives a better starting point
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    components = Vt[:n_components]
    
    max_time = 20  # seconds
    start_time = time.time()
    
    iteration = 0
    while iteration < max_iter:
        # Check timeout
        if time.time() - start_time > max_time:
            print(f"Sanger optimization timed out after {iteration} iterations")
            break
            
        # Adaptive learning rate
        learning_rate = initial_lr / (1 + lr_decay * iteration)
        
        # Process a batch of random samples
        batch_indices = np.random.choice(n_samples, min(batch_size, n_samples), replace=False)
        X_batch = X_centered[batch_indices]
        
        # Get projections
        Y = np.dot(X_batch, components.T)  # Shape: (batch_size, n_components)
        
        # Update each component
        for i in range(n_components):
            y_i = Y[:, i].reshape(-1, 1)
            
            if i > 0:
                inhibition_term = np.zeros_like(X_batch)
                for j in range(i):
                    y_j = Y[:, j].reshape(-1, 1)
                    inhibition_term += y_j * components[j].reshape(1, -1)
                
                update1 = np.mean(y_i * X_batch, axis=0)
                update2 = np.mean(y_i * inhibition_term, axis=0)
                delta = learning_rate * (update1 - update2)
                delta = np.clip(delta, -0.01, 0.01)
            else:
                update = np.mean(y_i * X_batch, axis=0)
                delta = learning_rate * update
                delta = np.clip(delta, -0.01, 0.01)
            
            components[i] += delta
        
        # Orthonormalize for stability
        components, _ = np.linalg.qr(components.T)
        components = components.T
        
        iteration += 1
    
    return components

# ------------------------------
# Helper functions for simulation
# ------------------------------
def generate_synthetic_data(n_normal=1000, n_intrusion=50, n_features=20):
    """
    Generate synthetic network data.
    Normal traffic is generated from a standard normal distribution.
    Intrusion traffic is generated from a shifted Gaussian distribution.
    """
    np.random.seed(42)  # for reproducibility
    
    normal_data = np.random.randn(n_normal, n_features)
    intrusion_data = np.random.randn(n_intrusion, n_features) + 3.0  # shift anomalies
    
    X = np.vstack([normal_data, intrusion_data])
    # Labels: 0 for normal, 1 for intrusion
    y = np.array([0] * n_normal + [1] * n_intrusion)
    
    # Shuffle the data
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    return X[indices], y[indices]

def reconstruction_error(X, components):
    """
    Given a PCA components matrix, compute the reconstruction error for each sample.
    
    Reconstruction is performed by projecting the centered data onto the components
    and then reconstructing from these projections.
    """
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    # Projection (coefficients for reconstruction)
    proj = np.dot(X_centered, components.T)
    # Reconstruct the original data
    X_reconstructed = np.dot(proj, components) + mean
    errors = np.linalg.norm(X - X_reconstructed, axis=1)
    return errors

# ...existing code...

def compute_metrics(errors, y, threshold=None):
    """
    Compute classification metrics for anomaly detection based on reconstruction errors.
    If threshold is None, it will be set automatically based on normal data.
    
    Returns confusion matrix values and threshold used.
    """
    # If no threshold provided, calculate one based on normal samples (e.g., 95th percentile)
    if threshold is None:
        normal_errors = errors[y == 0]
        threshold = np.percentile(normal_errors, 95)
    
    # Classify samples based on reconstruction error threshold
    predictions = (errors > threshold).astype(int)
    
    # Compute confusion matrix metrics
    tp = np.sum((predictions == 1) & (y == 1))  # True positives
    fp = np.sum((predictions == 1) & (y == 0))  # False positives
    tn = np.sum((predictions == 0) & (y == 0))  # True negatives
    fn = np.sum((predictions == 0) & (y == 1))  # False negatives
    
    # Derived metrics
    accuracy = (tp + tn) / len(y)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'accuracy': accuracy, 'precision': precision, 
        'recall': recall, 'f1': f1,
        'threshold': threshold,
        'predictions': predictions
    }

def evaluate_pca_methods(X, y, n_components=5):
    """
    Run all PCA methods on dataset X, compute the execution time and reconstruction errors.
    Returns a dictionary of results.
    """
    methods = {
        'SVD': pca_svd,
        'Krasulina': pca_krasulina,
        'Oja': pca_oja,
        'Sanger': pca_sanger,
        'Sanger Optimized': pca_sanger_optimized
    }
    
    results = {}
    
    for name, func in methods.items():
        start_time = time.time()
        components = func(X, n_components)
        elapsed = time.time() - start_time
        
        errors = reconstruction_error(X, components)
        avg_normal = np.mean(errors[y == 0])
        avg_intrusion = np.mean(errors[y == 1])
        
        # Compute classification metrics
        metrics = compute_metrics(errors, y)
        
        results[name] = {
            'time': elapsed,
            'avg_error_normal': avg_normal,
            'avg_error_intrusion': avg_intrusion,
            'errors': errors,  # full error vector for further analysis if desired
            'metrics': metrics  # classification metrics
        }
        
        print(f"Method: {name}")
        print(f"  Execution Time: {elapsed:.4f} sec")
        print(f"  Avg. Reconstruction Error (Normal): {avg_normal:.4f}")
        print(f"  Avg. Reconstruction Error (Intrusion): {avg_intrusion:.4f}")
        print("  Error Ratio (Intrusion/Normal): {:.4f}".format(avg_intrusion/avg_normal))
        print(f"  Threshold: {metrics['threshold']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")
        print(f"  Confusion Matrix:")
        print(f"    TP: {metrics['tp']} | FP: {metrics['fp']}")
        print(f"    FN: {metrics['fn']} | TN: {metrics['tn']}")
        print("-" * 50)
    
    return results

def plot_comparison(results):
    """
    Visualize reconstruction errors and confusion matrices from each PCA method.
    Save the visualization to the current directory.
    """
    method_names = list(results.keys())
    
    # Create a single large figure with a 2x3 grid
    fig = plt.figure(figsize=(18, 12))
    
    # Define grid positions
    ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=1)  # Reconstruction error
    ax2 = plt.subplot2grid((2, 3), (0, 1), colspan=2)  # Confusion matrix
    
    # Plot 1: Average reconstruction errors
    avg_normal = [results[m]['avg_error_normal'] for m in method_names]
    avg_intrusion = [results[m]['avg_error_intrusion'] for m in method_names]
    
    x = np.arange(len(method_names))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, avg_normal, width, label='Normal')
    bars2 = ax1.bar(x + width/2, avg_intrusion, width, label='Intrusion')
    
    ax1.set_ylabel('Average Reconstruction Error')
    ax1.set_title('Reconstruction Error Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(method_names, rotation=45)
    ax1.legend()
    
    for rect in bars1 + bars2:
        height = rect.get_height()
        ax1.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    # Plot 2: Confusion matrix visualization as a heatmap-style plot
    cm_data = np.zeros((len(method_names), 4))  # 4 values per method: TP, FP, FN, TN
    for i, method in enumerate(method_names):
        metrics = results[method]['metrics']
        cm_data[i, 0] = metrics['tp']
        cm_data[i, 1] = metrics['fp']
        cm_data[i, 2] = metrics['fn']
        cm_data[i, 3] = metrics['tn']
    
    im = ax2.imshow(cm_data, cmap='viridis')
    
    # Add labels and colorbars
    ax2.set_xticks(np.arange(4))
    ax2.set_xticklabels(['TP', 'FP', 'FN', 'TN'])
    ax2.set_yticks(np.arange(len(method_names)))
    ax2.set_yticklabels(method_names)
    ax2.set_title('Confusion Matrix Values')
    
    # Label values in the cells
    for i in range(len(method_names)):
        for j in range(4):
            ax2.text(j, i, f'{int(cm_data[i, j])}', 
                     ha='center', va='center', color='white' if cm_data[i, j] > np.max(cm_data)/2 else 'black')
    
    plt.colorbar(im, ax=ax2)
    
    # Add the detailed metrics in the bottom row
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
    
    for i, metric in enumerate(metrics_to_plot):
        ax = plt.subplot2grid((2, 3), (1, i if i < 3 else 2))
        values = [results[m]['metrics'][metric] for m in method_names]
        bars = ax.bar(np.arange(len(method_names)), values)
        ax.set_title(f'{metric.capitalize()}')
        ax.set_ylim(0, 1)
        ax.set_xticks(np.arange(len(method_names)))
        ax.set_xticklabels(method_names, rotation=45)
        
        # Add value labels
        for j, v in enumerate(values):
            ax.text(j, v + 0.02, f'{v:.2f}', ha='center')
    
    plt.tight_layout()
    
    # Save the figure to the current directory
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"./pca_comparison_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {filename}")
    
    plt.show()

# ...existing code...

# ------------------------------
# Main simulation
# ------------------------------
if __name__ == '__main__':
    # Generate synthetic network traffic data
    X, y = generate_synthetic_data(n_normal=1000, n_intrusion=50, n_features=20)
    
    # Evaluate all PCA methods
    results = evaluate_pca_methods(X, y, n_components=5)
    
    # Visualize the reconstruction error comparison
    plot_comparison(results)