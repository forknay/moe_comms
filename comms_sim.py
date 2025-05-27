import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
# ---------------------------
# Configuration Parameters
# ---------------------------

NUM_EXPERTS = 128            # Total number of experts in the MoE layer
NUM_GPUS = 8                 # Number of GPUs used in the system
EXPERTS_PER_GPU = NUM_EXPERTS // NUM_GPUS  # Experts evenly divided across GPUs
TOKENS = 1                # Number of tokens to simulate (set to 1 for decoder)
DTYPE_SIZE = 2               # Size in bytes (float16 = 2 bytes, float32 = 4 bytes)
NB_SHARED = 0                # Number of shared experts (NOT IMPLEMENTED)
TOP_K = 8                    # Number of routed experts assigned to each token
TOKEN_SIZE = 4096            # Embedding dimension size
TOT_EXPERTS = TOP_K + NB_SHARED  # Total experts activated per token

# ---------------------------
# Step 1: Generate Routing Table
# ---------------------------

def generate_routing(tokens, top_k, num_experts):
    """
    Randomly assign 'top_k' experts to each token.
    
    Returns a (tokens x top_k) matrix of expert indices.
    """
    return np.random.randint(0, num_experts, size=(tokens, top_k))


# ---------------------------
# Step 2: Map Each Expert to a GPU
# ---------------------------

def get_expert_gpu_map(num_experts, num_gpus):
    """
    Maps each expert to a GPU. Experts are evenly distributed across GPUs.
    
    Returns a dict: expert_id -> gpu_id
    """
    return {expert: expert // (num_experts // num_gpus) for expert in range(num_experts)}


# ---------------------------
# Step 3: Simulate All-to-All Communication
# ---------------------------

def simulate_all_to_all(routing, expert_gpu_map, num_gpus, token_size, dtype_size):
    """
    Simulates all-to-all communication volume between GPUs based on routing.
    
    Returns a (num_gpus x num_gpus) matrix where [i][j] = bytes sent from GPU i to GPU j.
    """
    comm_matrix = np.zeros((num_gpus, num_gpus))  # Communication matrix in bytes

    for token_id, experts in enumerate(routing):
        source_gpu = token_id % num_gpus  # Assign equal load to each GPU
        for expert in experts:
            target_gpu = expert_gpu_map[expert]
            if source_gpu != target_gpu:
                data_size = token_size * dtype_size
                comm_matrix[source_gpu][target_gpu] += data_size

    return comm_matrix


# ---------------------------
# Step 4: Visualize Communication Matrix
# ---------------------------

def plot_comm_matrix(matrix):
    """
    Plots the communication matrix using a heatmap (values in MB).
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix / (1024 ** 2), annot=True, fmt=".2f", cmap="Blues", square=True)
    plt.title("All-to-All Communication Load Between GPUs (MB)")
    plt.xlabel("Target GPU")
    plt.ylabel("Source GPU")
    plt.show()


# ---------------------------
# Run the Full Simulation
# ---------------------------

routing = generate_routing(TOKENS, TOP_K, NUM_EXPERTS)
expert_gpu_map = get_expert_gpu_map(NUM_EXPERTS, NUM_GPUS)
comm_matrix = simulate_all_to_all(routing, expert_gpu_map, NUM_GPUS, TOKEN_SIZE, DTYPE_SIZE)
plot_comm_matrix(comm_matrix)
