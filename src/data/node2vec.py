"""Use node2vec to generate node embeddings given an adjacency matrix A"""
import torch
import numpy as np
import torch_geometric
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.nn import Node2Vec
from scipy.sparse import csr_matrix

def generate_node_embeddings(adjacency_matrix, embedding_dim=10, epochs=100, 
                             walk_length=20, context_size=10, walks_per_node=5, 
                             num_negative_samples=1, p=2, q=0.5, lr=0.01):
    """
    Generate node embeddings using Node2Vec from an adjacency matrix.

    Parameters:
    - adjacency_matrix (np.ndarray): The adjacency matrix of the graph.
    - embedding_dim (int): The dimension of the node embeddings.
    - epochs (int): Number of training epochs.
    - walk_length (int): Length of each random walk.
    - context_size (int): Size of the context for training.
    - walks_per_node (int): Number of walks per node.
    - num_negative_samples (int): Number of negative samples.
    - p (float): Return parameter.
    - q (float): In-out parameter.
    - lr (float): Learning rate.

    Returns:
    - np.ndarray: The generated node embeddings.
    """

    # Convert the adjacency matrix to a sparse matrix
    sparse_A = csr_matrix(adjacency_matrix)

    # Convert to PyTorch Geometric data format
    edge_index, edge_attr = from_scipy_sparse_matrix(sparse_A)
    data = torch_geometric.data.Data(edge_index=edge_index, edge_attr=edge_attr)

    # Initialize Node2Vec
    node2vec = Node2Vec(data.edge_index, embedding_dim=embedding_dim, 
                        walk_length=walk_length, context_size=context_size, 
                        walks_per_node=walks_per_node, num_negative_samples=num_negative_samples, 
                        p=p, q=q, sparse=True)

    # Train Node2Vec
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    node2vec = node2vec.to(device)
    loader = node2vec.loader(batch_size=128,shuffle=True)
    # Optimizer
    optimizer = torch.optim.SparseAdam(node2vec.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        node2vec.train()
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = node2vec.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    # Get node embeddings
    node_embeddings = node2vec.embedding.weight.detach().cpu().numpy()
    node_embeddings = (node_embeddings -np.mean(node_embeddings))/(np.std(node_embeddings)+1e-8) #normalize the embeddings
    #clip
    node_embeddings = np.clip(node_embeddings, -4, 4) + np.random.normal(0, 0.1, node_embeddings.shape) #add noise to the embeddings
    
    return node_embeddings