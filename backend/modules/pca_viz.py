"""
PCA Visualization Module
Generates data for Principal Component Analysis visualizations
"""

import numpy as np
from typing import Dict, List, Any, Tuple


def generate_2d_data(data_type: str, n_samples: int = 60) -> Dict[str, Any]:
    """
    Generate 2D data for PCA projection visualization.
    
    Args:
        data_type: 'clusters', 'linear', or 'random'
        n_samples: Number of data points
    
    Returns:
        Dictionary with data points and PCA results
    """
    np.random.seed(42)
    
    if data_type == 'clusters':
        # Two distinct clusters
        n_per_cluster = n_samples // 2
        
        cluster1_x = np.random.randn(n_per_cluster) * 1.5 - 3
        cluster1_y = np.random.randn(n_per_cluster) * 1.5 + 2
        
        cluster2_x = np.random.randn(n_per_cluster) * 1.5 + 3
        cluster2_y = np.random.randn(n_per_cluster) * 1.5 - 2
        
        X = np.concatenate([cluster1_x, cluster2_x])
        Y = np.concatenate([cluster1_y, cluster2_y])
        labels = [0] * n_per_cluster + [1] * n_per_cluster
        
    elif data_type == 'linear':
        # Linear correlation with noise
        t = np.linspace(-4, 4, n_samples)
        X = t + np.random.randn(n_samples) * 0.5
        Y = t * 0.7 + np.random.randn(n_samples) * 0.8
        labels = [0] * n_samples
        
    else:  # random
        X = np.random.randn(n_samples) * 2
        Y = np.random.randn(n_samples) * 2
        labels = [0] * n_samples
    
    # Center the data
    mean_x, mean_y = np.mean(X), np.mean(Y)
    X_centered = X - mean_x
    Y_centered = Y - mean_y
    
    # Compute covariance matrix
    cov_xx = np.mean(X_centered ** 2)
    cov_yy = np.mean(Y_centered ** 2)
    cov_xy = np.mean(X_centered * Y_centered)
    
    # Compute eigenvalues and eigenvectors
    # For 2x2 symmetric matrix: λ² - (a+d)λ + (ad-bc) = 0
    trace = cov_xx + cov_yy
    det = cov_xx * cov_yy - cov_xy ** 2
    
    discriminant = trace ** 2 / 4 - det
    if discriminant < 0:
        discriminant = 0
    
    lambda1 = trace / 2 + np.sqrt(discriminant)
    lambda2 = trace / 2 - np.sqrt(discriminant)
    
    # Compute eigenvector for PC1
    if abs(cov_xy) > 1e-10:
        pc1_x = lambda1 - cov_yy
        pc1_y = cov_xy
    else:
        if cov_xx > cov_yy:
            pc1_x, pc1_y = 1, 0
        else:
            pc1_x, pc1_y = 0, 1
    
    # Normalize PC1
    norm = np.sqrt(pc1_x ** 2 + pc1_y ** 2)
    pc1_x, pc1_y = pc1_x / norm, pc1_y / norm
    
    # PC2 is perpendicular to PC1
    pc2_x, pc2_y = -pc1_y, pc1_x
    
    # Compute variance explained
    total_variance = lambda1 + lambda2
    variance_pc1 = lambda1 / total_variance * 100
    variance_pc2 = lambda2 / total_variance * 100
    
    # Project data onto PC1
    projected_x = (X_centered * pc1_x + Y_centered * pc1_y) * pc1_x + mean_x
    projected_y = (X_centered * pc1_x + Y_centered * pc1_y) * pc1_y + mean_y
    
    # Scale for visualization (canvas coordinates)
    scale = 50
    center_x, center_y = 300, 200
    
    data_points = []
    projected_points = []
    
    for i in range(len(X)):
        data_points.append({
            'x': center_x + X[i] * scale,
            'y': center_y - Y[i] * scale,  # Flip Y for canvas
            'label': labels[i]
        })
        projected_points.append({
            'x': center_x + (projected_x[i] - mean_x) * scale + mean_x * scale,
            'y': center_y - (projected_y[i] - mean_y) * scale - mean_y * scale
        })
    
    return {
        'data_points': data_points,
        'projected_points': projected_points,
        'pc1': {
            'x': pc1_x,
            'y': pc1_y,
            'variance': round(variance_pc1, 1)
        },
        'pc2': {
            'x': pc2_x,
            'y': pc2_y,
            'variance': round(variance_pc2, 1)
        },
        'total_variance': round(variance_pc1 + variance_pc2, 1),
        'center': {'x': center_x, 'y': center_y},
        'scale': scale
    }


def generate_scree_data(num_features: int, data_type: str) -> Dict[str, Any]:
    """
    Generate scree plot data for variance explained visualization.
    
    Args:
        num_features: Number of original features
        data_type: 'structured', 'moderate', or 'random'
    
    Returns:
        Dictionary with eigenvalues and cumulative variance
    """
    np.random.seed(42)
    
    if data_type == 'structured':
        # First few components explain most variance
        base_eigenvalues = np.array([5.0, 3.0, 1.5, 0.8, 0.5, 0.3, 0.2, 0.15, 0.1, 0.05])
        
    elif data_type == 'moderate':
        # More gradual decline
        base_eigenvalues = np.array([3.0, 2.5, 2.0, 1.5, 1.2, 0.9, 0.7, 0.5, 0.4, 0.3])
        
    else:  # random
        # Nearly equal eigenvalues
        base_eigenvalues = np.array([1.3, 1.2, 1.1, 1.05, 1.0, 0.95, 0.9, 0.85, 0.8, 0.75])
    
    # Adjust to requested number of features
    if num_features <= 10:
        eigenvalues = base_eigenvalues[:num_features]
    else:
        # Extend with small values
        eigenvalues = np.concatenate([
            base_eigenvalues,
            np.linspace(0.05, 0.02, num_features - 10)
        ])
    
    # Normalize to sum to num_features (each original feature has variance 1)
    eigenvalues = eigenvalues * num_features / np.sum(eigenvalues)
    
    # Calculate variance explained
    total_variance = np.sum(eigenvalues)
    variance_explained = eigenvalues / total_variance * 100
    cumulative_variance = np.cumsum(variance_explained)
    
    components = []
    for i in range(num_features):
        components.append({
            'component': i + 1,
            'eigenvalue': round(eigenvalues[i], 3),
            'variance_explained': round(variance_explained[i], 1),
            'cumulative_variance': round(cumulative_variance[i], 1)
        })
    
    return {
        'components': components,
        'num_features': num_features,
        'data_type': data_type
    }


def get_chemistry_pca_data(dataset: str) -> Dict[str, Any]:
    """
    Generate PCA visualization for chemistry datasets.
    
    Args:
        dataset: 'drug', 'solvents', or 'elements'
    
    Returns:
        Dictionary with PCA-projected chemistry data
    """
    np.random.seed(42)
    
    if dataset == 'drug':
        # Drug molecules with different properties
        molecules = [
            {'name': 'Aspirin', 'class': 'NSAID', 'mw': 180, 'logp': 1.2, 'tpsa': 63},
            {'name': 'Ibuprofen', 'class': 'NSAID', 'mw': 206, 'logp': 3.5, 'tpsa': 37},
            {'name': 'Naproxen', 'class': 'NSAID', 'mw': 230, 'logp': 3.3, 'tpsa': 37},
            {'name': 'Diclofenac', 'class': 'NSAID', 'mw': 296, 'logp': 4.5, 'tpsa': 49},
            {'name': 'Paracetamol', 'class': 'Analgesic', 'mw': 151, 'logp': 0.5, 'tpsa': 49},
            {'name': 'Morphine', 'class': 'Opioid', 'mw': 285, 'logp': 0.9, 'tpsa': 52},
            {'name': 'Codeine', 'class': 'Opioid', 'mw': 299, 'logp': 1.2, 'tpsa': 41},
            {'name': 'Fentanyl', 'class': 'Opioid', 'mw': 336, 'logp': 4.1, 'tpsa': 23},
            {'name': 'Caffeine', 'class': 'Stimulant', 'mw': 194, 'logp': -0.1, 'tpsa': 58},
            {'name': 'Nicotine', 'class': 'Stimulant', 'mw': 162, 'logp': 1.2, 'tpsa': 25},
            {'name': 'Atorvastatin', 'class': 'Statin', 'mw': 558, 'logp': 4.3, 'tpsa': 111},
            {'name': 'Simvastatin', 'class': 'Statin', 'mw': 418, 'logp': 4.7, 'tpsa': 72},
            {'name': 'Metformin', 'class': 'Antidiabetic', 'mw': 129, 'logp': -1.4, 'tpsa': 91},
            {'name': 'Glipizide', 'class': 'Antidiabetic', 'mw': 445, 'logp': 2.0, 'tpsa': 109},
            {'name': 'Amoxicillin', 'class': 'Antibiotic', 'mw': 365, 'logp': 0.0, 'tpsa': 132},
            {'name': 'Penicillin G', 'class': 'Antibiotic', 'mw': 334, 'logp': 1.5, 'tpsa': 97},
        ]
        
        class_colors = {
            'NSAID': '#3b82f6',
            'Analgesic': '#10b981',
            'Opioid': '#7c3aed',
            'Stimulant': '#f59e0b',
            'Statin': '#ef4444',
            'Antidiabetic': '#06b6d4',
            'Antibiotic': '#ec4899'
        }
        
    elif dataset == 'solvents':
        # Common solvents
        molecules = [
            {'name': 'Water', 'class': 'Protic', 'mw': 18, 'logp': -0.8, 'tpsa': 25},
            {'name': 'Methanol', 'class': 'Protic', 'mw': 32, 'logp': -0.7, 'tpsa': 20},
            {'name': 'Ethanol', 'class': 'Protic', 'mw': 46, 'logp': -0.3, 'tpsa': 20},
            {'name': 'Isopropanol', 'class': 'Protic', 'mw': 60, 'logp': 0.3, 'tpsa': 20},
            {'name': 'Acetic Acid', 'class': 'Protic', 'mw': 60, 'logp': -0.2, 'tpsa': 37},
            {'name': 'DMSO', 'class': 'Aprotic', 'mw': 78, 'logp': -1.4, 'tpsa': 37},
            {'name': 'DMF', 'class': 'Aprotic', 'mw': 73, 'logp': -1.0, 'tpsa': 20},
            {'name': 'Acetone', 'class': 'Aprotic', 'mw': 58, 'logp': -0.2, 'tpsa': 17},
            {'name': 'THF', 'class': 'Aprotic', 'mw': 72, 'logp': 0.5, 'tpsa': 12},
            {'name': 'Dichloromethane', 'class': 'Halogenated', 'mw': 85, 'logp': 1.3, 'tpsa': 0},
            {'name': 'Chloroform', 'class': 'Halogenated', 'mw': 119, 'logp': 1.9, 'tpsa': 0},
            {'name': 'Toluene', 'class': 'Aromatic', 'mw': 92, 'logp': 2.7, 'tpsa': 0},
            {'name': 'Benzene', 'class': 'Aromatic', 'mw': 78, 'logp': 2.1, 'tpsa': 0},
            {'name': 'Hexane', 'class': 'Alkane', 'mw': 86, 'logp': 3.6, 'tpsa': 0},
            {'name': 'Cyclohexane', 'class': 'Alkane', 'mw': 84, 'logp': 3.0, 'tpsa': 0},
        ]
        
        class_colors = {
            'Protic': '#3b82f6',
            'Aprotic': '#10b981',
            'Halogenated': '#f59e0b',
            'Aromatic': '#7c3aed',
            'Alkane': '#ef4444'
        }
        
    else:  # elements
        # Chemical elements
        molecules = [
            {'name': 'H', 'class': 'Nonmetal', 'mw': 1, 'logp': 0, 'tpsa': 0},
            {'name': 'C', 'class': 'Nonmetal', 'mw': 12, 'logp': 0, 'tpsa': 0},
            {'name': 'N', 'class': 'Nonmetal', 'mw': 14, 'logp': 0, 'tpsa': 0},
            {'name': 'O', 'class': 'Nonmetal', 'mw': 16, 'logp': 0, 'tpsa': 0},
            {'name': 'F', 'class': 'Halogen', 'mw': 19, 'logp': 0, 'tpsa': 0},
            {'name': 'Cl', 'class': 'Halogen', 'mw': 35, 'logp': 0, 'tpsa': 0},
            {'name': 'Br', 'class': 'Halogen', 'mw': 80, 'logp': 0, 'tpsa': 0},
            {'name': 'I', 'class': 'Halogen', 'mw': 127, 'logp': 0, 'tpsa': 0},
            {'name': 'Na', 'class': 'Metal', 'mw': 23, 'logp': 0, 'tpsa': 0},
            {'name': 'K', 'class': 'Metal', 'mw': 39, 'logp': 0, 'tpsa': 0},
            {'name': 'Mg', 'class': 'Metal', 'mw': 24, 'logp': 0, 'tpsa': 0},
            {'name': 'Ca', 'class': 'Metal', 'mw': 40, 'logp': 0, 'tpsa': 0},
            {'name': 'Fe', 'class': 'Transition', 'mw': 56, 'logp': 0, 'tpsa': 0},
            {'name': 'Cu', 'class': 'Transition', 'mw': 64, 'logp': 0, 'tpsa': 0},
            {'name': 'Zn', 'class': 'Transition', 'mw': 65, 'logp': 0, 'tpsa': 0},
            {'name': 'Pd', 'class': 'Transition', 'mw': 106, 'logp': 0, 'tpsa': 0},
            {'name': 'Pt', 'class': 'Transition', 'mw': 195, 'logp': 0, 'tpsa': 0},
            {'name': 'Au', 'class': 'Transition', 'mw': 197, 'logp': 0, 'tpsa': 0},
        ]
        
        class_colors = {
            'Nonmetal': '#3b82f6',
            'Halogen': '#10b981',
            'Metal': '#f59e0b',
            'Transition': '#7c3aed'
        }
    
    # Extract features for PCA
    features = np.array([[m['mw'], m['logp'], m['tpsa']] for m in molecules])
    
    # Standardize features
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    std[std == 0] = 1  # Avoid division by zero
    features_std = (features - mean) / std
    
    # Compute covariance matrix
    cov_matrix = np.cov(features_std.T)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort by eigenvalue (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Project data onto first two principal components
    projected = features_std @ eigenvectors[:, :2]
    
    # Scale for visualization
    scale = 40
    center_x, center_y = 300, 200
    
    # Normalize to [-1, 1] range then scale
    projected_norm = (projected - projected.mean(axis=0)) / (projected.max(axis=0) - projected.min(axis=0) + 0.1)
    
    points = []
    classes_found = set()
    
    for i, mol in enumerate(molecules):
        points.append({
            'name': mol['name'],
            'class': mol['class'],
            'x': center_x + projected_norm[i, 0] * scale * 3,
            'y': center_y - projected_norm[i, 1] * scale * 3,
            'color': class_colors.get(mol['class'], '#6b7280')
        })
        classes_found.add(mol['class'])
    
    # Calculate variance explained
    total_variance = np.sum(eigenvalues)
    variance_pc1 = eigenvalues[0] / total_variance * 100
    variance_pc2 = eigenvalues[1] / total_variance * 100
    
    # Build legend
    legend = []
    for cls, color in class_colors.items():
        if cls in classes_found:
            legend.append({'class': cls, 'color': color})
    
    return {
        'name': dataset.title() + ' Dataset',
        'points': points,
        'legend': legend,
        'pc1_variance': round(variance_pc1, 1),
        'pc2_variance': round(variance_pc2, 1),
        'total_variance': round(variance_pc1 + variance_pc2, 1),
        'features_used': ['Molecular Weight', 'LogP', 'TPSA']
    }
