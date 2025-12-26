"""
simulate.py

Responsible for generating simualted graphs based on a certain experiment spec

"""


import os
import json
import time
import random
from typing import Dict, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

# ----------------------------
# Main run function
# ----------------------------

def run_simulation(spec: Dict) -> str:
    """
    Runs the full simulation pipeline.

    Args:
        spec (dict): Experiment specification loaded from YAML

    Returns:
        output_dir (str): Path to the run directory
    """
    exp = spec["experiment"]
    seed = exp.get("seed", 42)
    base_output = exp.get("output_dir", "runs")

    set_seed(seed)
    output_dir = create_output_dir(base_output)

    #Graph Generation
    edge_index = generate_graph(spec)
    num_nodes = spec["graph"]["num_nodes"]

    x = generate_features(spec, num_nodes)
    y = generate_labels(spec, num_nodes)

    #this is the simulated data we make for the GNN
    data = Data(
        x=x,
        edge_index=edge_index,
        y=y
    )

    
    stats = compute_stats(data)

    
    torch.save(data, os.path.join(output_dir, "data.pt"))

    with open(os.path.join(output_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    return output_dir


# ----------------------------
# Graph generation
# ----------------------------

def generate_graph(spec: Dict) -> torch.Tensor:
    """
    Generates graph structure (edge_index).

    Args:
        spec (dict): Experiment specification

    Returns:
        edge_index (torch.Tensor): Shape [2, num_edges]
    """
    graph_spec = spec["graph"]
    num_nodes = graph_spec["num_nodes"]
    avg_degree = graph_spec.get("avg_degree", 4)
    directed = graph_spec.get("directed", False)

    # Convert average degree to edge probability
    p = avg_degree / (num_nodes - 1)

    edges = []

    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if random.random() < p:
                edges.append((i,j))
                if not directed:
                    edges.append((j,i))
    
    if len(edges) == 0:
        raise ValueError("Generated graph has no edges. Check avg_degree.")
    
    edge_index = torch.tensor(edges, dtype=torch.long).t()

    return edge_index

# ----------------------------
# Feature generation
# ----------------------------

def generate_features(spec: Dict, num_nodes: int) -> torch.Tensor:
    """
    Generates node features.

    Args:
        spec (dict): Experiment specification
        num_nodes (int): Number of nodes in the graph

    Returns:
        x (torch.Tensor): Node feature matrix [num_nodes, num_features]
    """
    features = spec.get("features",{})

    feature_cols= []

    #Numeric

    for ft in features.get("numeric", []):
        dist = ft.get("distribution", "normal")

        if dist == "normal":
            values = np.random.normal(loc=0.0, scale=1.0, size=num_nodes)
        elif dist == "lognormal":
            values = np.random.lognormal(mean=0.0, sigma=1.0, size=num_nodes)
        else:
            raise ValueError(f"Unknown numeric distribution: {dist}")

        feature_cols.append(values.reshape(-1, 1))

    #Categorical

    for ft in features.get("categorical", []):
        categories = ft["values"]
        num_categories = len(categories)

        values = np.random.randint(
            low=0,
            high=num_categories,
            size=num_nodes
        )

        feature_cols.append(values.reshape(-1, 1))

    if not feature_cols:
        raise ValueError("No features specified in spec.")

    # Concatenate all feature columns
    x = np.hstack(feature_cols)

    return torch.tensor(x, dtype=torch.float)

# ----------------------------
# Label generation
# ----------------------------

def generate_labels(spec: Dict, num_nodes: int) -> torch.Tensor:
    """
    Generates node labels.

    Args:
        spec (dict): Experiment specification
        num_nodes (int): Number of nodes

    Returns:
        y (torch.Tensor): Labels tensor [num_nodes]
    """
    label = spec.get("labels", {})
    pos_rate = label.get("positive_class_rate", 0.5)

    if not (0.0 < pos_rate < 1.0):
        raise ValueError("positive_class_rate must be between 0 and 1")

    num_pos = int(num_nodes * pos_rate)

    y = np.zeros(num_nodes, dtype=np.int64)

    pos_indices = np.random.choice(
        num_nodes,
        size=num_pos,
        replace=False
    )

    y[pos_indices] = 1

    return torch.tensor(y, dtype=torch.long)


# ----------------------------
# Dataset statistics
# ----------------------------

def compute_stats(data: Data) -> Dict:
    """
    Computes basic dataset statistics.

    Args:
        data (Data): PyG Data object

    Returns:
        stats (dict): Dataset summary
    """
    num_nodes = data.num_nodes
    num_edges = data.edge_index.size(1)

    stats = {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "num_features": data.x.size(1) if data.x is not None else 0
    }

    if data.y is not None:
        y = data.y
        stats["label_distribution"] = {
            "positive_rate": y.float().mean().item(),
            "num_positive": int((y == 1).sum().item()),
            "num_negative": int((y == 0).sum().item())
        }

    # Average degree (for directed graphs, this is out-degree)
    stats["avg_degree"] = num_edges / num_nodes

    return stats

# ----------------------------
# Utility helpers
# ----------------------------

def set_seed(seed: int):
    """
    Sets random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def create_output_dir(base_dir: str) -> str:
    """
    Creates a timestamped output directory.

    Args:
        base_dir (str): Base runs directory

    Returns:
        output_dir (str)
    """
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(base_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir



