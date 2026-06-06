"""
GNN Visualization Module
Generates data for Graph Neural Network visualizations
"""

import numpy as np
from typing import Dict, List, Any

ELEMENT_COLORS = {
    'H': '#60a5fa',
    'C': '#374151',
    'N': '#3b82f6',
    'O': '#ef4444',
    'S': '#f59e0b',
    'P': '#7c3aed',
    'F': '#10b981',
    'Cl': '#10b981',
    'Br': '#a16207',
    'I': '#7c3aed',
}

ATOM_FEATURES = {
    'H': {'atomic_num': 1, 'degree': 1, 'hybridization': 's', 'electronegativity': 2.2},
    'C': {'atomic_num': 6, 'degree': 4, 'hybridization': 'sp3', 'electronegativity': 2.55},
    'N': {'atomic_num': 7, 'degree': 3, 'hybridization': 'sp3', 'electronegativity': 3.04},
    'O': {'atomic_num': 8, 'degree': 2, 'hybridization': 'sp3', 'electronegativity': 3.44},
    'S': {'atomic_num': 16, 'degree': 2, 'hybridization': 'sp3', 'electronegativity': 2.58},
    'P': {'atomic_num': 15, 'degree': 3, 'hybridization': 'sp3', 'electronegativity': 2.19},
    'F': {'atomic_num': 9, 'degree': 1, 'hybridization': 's', 'electronegativity': 3.98},
    'Cl': {'atomic_num': 17, 'degree': 1, 'hybridization': 's', 'electronegativity': 3.16},
    'Br': {'atomic_num': 35, 'degree': 1, 'hybridization': 's', 'electronegativity': 2.96},
    'I': {'atomic_num': 53, 'degree': 1, 'hybridization': 's', 'electronegativity': 2.66},
}


def generate_sample_graph(num_nodes: int = 6) -> Dict[str, Any]:
    np.random.seed(42)
    nodes = []
    center_x, center_y = 300, 200
    radius = 120

    for i in range(num_nodes):
        angle = (i / num_nodes) * 2 * np.pi - np.pi / 2
        nodes.append({
            'id': i,
            'x': center_x + radius * np.cos(angle),
            'y': center_y + radius * np.sin(angle),
            'feature': round(np.random.random(), 3),
            'state': 'default',
            'message': 0
        })
    edges = []
    for i in range(num_nodes):
        edges.append({
            'source': i,
            'target': (i + 1) % num_nodes,
            'type': 'ring'
        })
        if i < num_nodes // 2:
            edges.append({
                'source': i,
                'target': i + num_nodes // 2,
                'type': 'cross'
            })

    adjacency = {i: [] for i in range(num_nodes)}
    for edge in edges:
        adjacency[edge['source']].append(edge['target'])
        adjacency[edge['target']].append(edge['source'])

    for i in range(num_nodes):
        nodes[i]['neighbors'] = adjacency[i]

    return {
        'nodes': nodes,
        'edges': edges,
        'adjacency': adjacency
    }


def simulate_message_passing(nodes: List[Dict], edges: List[Dict],
                            current_step: int) -> Dict[str, Any]:
    num_nodes = len(nodes)

    if current_step >= num_nodes:
        for node in nodes:
            node['state'] = 'default'
            node['feature'] = round(np.random.random(), 3)
        return {
            'nodes': nodes,
            'edges': edges,
            'message': None,
            'complete': True
        }
    for edge in edges:
        edge['active'] = False
    target_node = nodes[current_step]
    target_node['state'] = 'processing'

    neighbor_features = []
    message_value = 0

    for neighbor_id in target_node['neighbors']:
        neighbor = nodes[neighbor_id]
        neighbor_features.append(neighbor['feature'])
        message_value += neighbor['feature']

        for edge in edges:
            if (edge['source'] == current_step and edge['target'] == neighbor_id) or \
               (edge['target'] == current_step and edge['source'] == neighbor_id):
                edge['active'] = True

    new_feature = message_value / len(target_node['neighbors']) if target_node['neighbors'] else 0

    message_info = {
        'node_id': current_step,
        'neighbors': target_node['neighbors'],
        'neighbor_features': neighbor_features,
        'message_sum': round(message_value, 3),
        'new_feature': round(new_feature, 3)
    }

    target_node['feature'] = round(new_feature, 3)
    target_node['state'] = 'updated'

    return {
        'nodes': nodes,
        'edges': edges,
        'message': message_info,
        'complete': False
    }


def get_molecule_data(molecule_type: str) -> Dict[str, Any]:
    center_x, center_y = 300, 175

    if molecule_type == 'benzene':
        radius = 70
        atoms = []
        bonds = []
        for i in range(6):
            angle = (i / 6) * 2 * np.pi - np.pi / 2
            atoms.append({
                'id': i,
                'element': 'C',
                'x': center_x + radius * np.cos(angle),
                'y': center_y + radius * np.sin(angle),
                'color': ELEMENT_COLORS['C'],
                'features': ATOM_FEATURES['C']
            })
        for i in range(6):
            angle = (i / 6) * 2 * np.pi - np.pi / 2
            atoms.append({
                'id': i + 6,
                'element': 'H',
                'x': center_x + (radius + 50) * np.cos(angle),
                'y': center_y + (radius + 50) * np.sin(angle),
                'color': ELEMENT_COLORS['H'],
                'features': ATOM_FEATURES['H']
            })

        for i in range(6):
            bonds.append({
                'source': i,
                'target': (i + 1) % 6,
                'type': 'aromatic',
                'color': '#10b981'
            })
            bonds.append({
                'source': i,
                'target': i + 6,
                'type': 'single',
                'color': '#3b82f6'
            })

        return {
            'name': 'Benzene',
            'formula': 'C₆H₆',
            'atoms': atoms,
            'bonds': bonds,
            'properties': {
                'molecular_weight': 78.11,
                'logP': 2.13,
                'tpsa': 0,
                'hbd': 0,
                'hba': 0
            }
        }

    elif molecule_type == 'ethanol':
        atoms = [
            {'id': 0, 'element': 'C', 'x': center_x - 80, 'y': center_y,
             'color': ELEMENT_COLORS['C'], 'features': ATOM_FEATURES['C']},
            {'id': 1, 'element': 'C', 'x': center_x, 'y': center_y,
             'color': ELEMENT_COLORS['C'], 'features': ATOM_FEATURES['C']},
            {'id': 2, 'element': 'O', 'x': center_x + 80, 'y': center_y,
             'color': ELEMENT_COLORS['O'], 'features': ATOM_FEATURES['O']},
            {'id': 3, 'element': 'H', 'x': center_x + 120, 'y': center_y - 20,
             'color': ELEMENT_COLORS['H'], 'features': ATOM_FEATURES['H']},
            {'id': 4, 'element': 'H', 'x': center_x - 120, 'y': center_y - 30,
             'color': ELEMENT_COLORS['H'], 'features': ATOM_FEATURES['H']},
            {'id': 5, 'element': 'H', 'x': center_x - 100, 'y': center_y + 40,
             'color': ELEMENT_COLORS['H'], 'features': ATOM_FEATURES['H']},
            {'id': 6, 'element': 'H', 'x': center_x - 60, 'y': center_y - 50,
             'color': ELEMENT_COLORS['H'], 'features': ATOM_FEATURES['H']},
            {'id': 7, 'element': 'H', 'x': center_x + 20, 'y': center_y - 40,
             'color': ELEMENT_COLORS['H'], 'features': ATOM_FEATURES['H']},
            {'id': 8, 'element': 'H', 'x': center_x + 20, 'y': center_y + 40,
             'color': ELEMENT_COLORS['H'], 'features': ATOM_FEATURES['H']}
        ]

        bonds = [
            {'source': 0, 'target': 1, 'type': 'single', 'color': '#3b82f6'},
            {'source': 1, 'target': 2, 'type': 'single', 'color': '#3b82f6'},
            {'source': 2, 'target': 3, 'type': 'single', 'color': '#3b82f6'},
            {'source': 0, 'target': 4, 'type': 'single', 'color': '#3b82f6'},
            {'source': 0, 'target': 5, 'type': 'single', 'color': '#3b82f6'},
            {'source': 0, 'target': 6, 'type': 'single', 'color': '#3b82f6'},
            {'source': 1, 'target': 7, 'type': 'single', 'color': '#3b82f6'},
            {'source': 1, 'target': 8, 'type': 'single', 'color': '#3b82f6'}
        ]

        return {
            'name': 'Ethanol',
            'formula': 'C₂H₅OH',
            'atoms': atoms,
            'bonds': bonds,
            'properties': {
                'molecular_weight': 46.07,
                'logP': -0.31,
                'tpsa': 20.2,
                'hbd': 1,
                'hba': 1
            }
        }

    elif molecule_type == 'caffeine':
        scale = 0.8
        atoms = [
            {'id': 0, 'element': 'N', 'x': center_x - 60 * scale, 'y': center_y - 60 * scale,
             'color': ELEMENT_COLORS['N'], 'features': ATOM_FEATURES['N']},
            {'id': 1, 'element': 'C', 'x': center_x, 'y': center_y - 80 * scale,
             'color': ELEMENT_COLORS['C'], 'features': ATOM_FEATURES['C']},
            {'id': 2, 'element': 'N', 'x': center_x + 60 * scale, 'y': center_y - 60 * scale,
             'color': ELEMENT_COLORS['N'], 'features': ATOM_FEATURES['N']},
            {'id': 3, 'element': 'C', 'x': center_x + 70 * scale, 'y': center_y,
             'color': ELEMENT_COLORS['C'], 'features': ATOM_FEATURES['C']},
            {'id': 4, 'element': 'C', 'x': center_x + 60 * scale, 'y': center_y + 60 * scale,
             'color': ELEMENT_COLORS['C'], 'features': ATOM_FEATURES['C']},
            {'id': 5, 'element': 'N', 'x': center_x, 'y': center_y + 70 * scale,
             'color': ELEMENT_COLORS['N'], 'features': ATOM_FEATURES['N']},
            {'id': 6, 'element': 'C', 'x': center_x - 60 * scale, 'y': center_y + 50 * scale,
             'color': ELEMENT_COLORS['C'], 'features': ATOM_FEATURES['C']},
            {'id': 7, 'element': 'C', 'x': center_x - 80 * scale, 'y': center_y,
             'color': ELEMENT_COLORS['C'], 'features': ATOM_FEATURES['C']},
            {'id': 8, 'element': 'O', 'x': center_x, 'y': center_y - 130 * scale,
             'color': ELEMENT_COLORS['O'], 'features': ATOM_FEATURES['O']},
            {'id': 9, 'element': 'O', 'x': center_x + 60 * scale, 'y': center_y + 110 * scale,
             'color': ELEMENT_COLORS['O'], 'features': ATOM_FEATURES['O']},
        ]

        bonds = [
            {'source': 0, 'target': 1, 'type': 'single', 'color': '#3b82f6'},
            {'source': 1, 'target': 2, 'type': 'single', 'color': '#3b82f6'},
            {'source': 2, 'target': 3, 'type': 'single', 'color': '#3b82f6'},
            {'source': 3, 'target': 4, 'type': 'double', 'color': '#ef4444'},
            {'source': 4, 'target': 5, 'type': 'single', 'color': '#3b82f6'},
            {'source': 5, 'target': 6, 'type': 'single', 'color': '#3b82f6'},
            {'source': 6, 'target': 7, 'type': 'double', 'color': '#ef4444'},
            {'source': 7, 'target': 0, 'type': 'single', 'color': '#3b82f6'},
            {'source': 1, 'target': 8, 'type': 'double', 'color': '#ef4444'},
            {'source': 4, 'target': 9, 'type': 'double', 'color': '#ef4444'},
        ]

        return {
            'name': 'Caffeine',
            'formula': 'C₈H₁₀N₄O₂',
            'atoms': atoms,
            'bonds': bonds,
            'properties': {
                'molecular_weight': 194.19,
                'logP': -0.07,
                'tpsa': 58.4,
                'hbd': 0,
                'hba': 6
            }
        }

    else:
        return get_molecule_data('benzene')


def get_gnn_embedding_demo() -> Dict[str, Any]:
    np.random.seed(42)
    num_nodes_per_class = 10
    num_classes = 3

    layers = []

    layer0 = []
    for c in range(num_classes):
        for i in range(num_nodes_per_class):
            layer0.append({
                'id': c * num_nodes_per_class + i,
                'class': c,
                'x': np.random.randn() * 2,
                'y': np.random.randn() * 2
            })
    layers.append({'layer': 0, 'name': 'Input Features', 'embeddings': layer0})

    layer1 = []
    centers = [(0, 2), (-1.7, -1), (1.7, -1)]
    for c in range(num_classes):
        for i in range(num_nodes_per_class):
            layer1.append({
                'id': c * num_nodes_per_class + i,
                'class': c,
                'x': centers[c][0] + np.random.randn() * 0.8,
                'y': centers[c][1] + np.random.randn() * 0.8
            })
    layers.append({'layer': 1, 'name': 'After Layer 1', 'embeddings': layer1})

    layer2 = []
    for c in range(num_classes):
        for i in range(num_nodes_per_class):
            layer2.append({
                'id': c * num_nodes_per_class + i,
                'class': c,
                'x': centers[c][0] * 1.5 + np.random.randn() * 0.4,
                'y': centers[c][1] * 1.5 + np.random.randn() * 0.4
            })
    layers.append({'layer': 2, 'name': 'After Layer 2', 'embeddings': layer2})

    layer3 = []
    for c in range(num_classes):
        for i in range(num_nodes_per_class):
            layer3.append({
                'id': c * num_nodes_per_class + i,
                'class': c,
                'x': centers[c][0] * 2 + np.random.randn() * 0.2,
                'y': centers[c][1] * 2 + np.random.randn() * 0.2
            })
    layers.append({'layer': 3, 'name': 'Final Embeddings', 'embeddings': layer3})

    return {
        'layers': layers,
        'class_names': ['Class A (Active)', 'Class B (Moderate)', 'Class C (Inactive)'],
        'class_colors': ['#3b82f6', '#10b981', '#ef4444']
    }
