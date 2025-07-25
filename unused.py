import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# ---------------------------
# Configuration Parameters
# ---------------------------

# 8 DP 8 EP, one gpu per node
NUM_EXPERTS = 128            # Total number of experts in the MoE layer
NUM_GPUS = 8                 # Number of GPUs used in the system
EXPERTS_PER_GPU = NUM_EXPERTS // NUM_GPUS  # Experts evenly divided across GPUs

TOKENS = 1024                # Number of tokens to simulate (set to 1 for decoder, 1024 for encoder)
TOKENS_PER_GPU = TOKENS // NUM_GPUS  # Tokens evenly divided across GPUs
DTYPE_SIZE = 4               # Size in bytes (float16 = 2 bytes, float32 = 4 bytes)
TOP_K_EXPERTS = 8           # Number of routed experts assigned to each token
TOP_K_NODES = 4        # Number of nodes to route to (for each token)
EMBED_DIM = 2048            # Embedding dimension size


IS_BALANCED = False         # Set to False for imbalanced routing
HOT_RATIO = 0.5            # Ratio of hot experts (for imbalanced routing), 1.0 for balanced
HOT_WEIGHT = 0.9            # Weight for hot experts in imbalanced routing, 1.0 for balanced

if NUM_EXPERTS % NUM_GPUS != 0:
    raise ValueError("NUM_EXPERTS must be divisible by NUM_GPUS for even distribution of experts across GPUs.")
# ---------------------------
# Step 1: Generate Routing Table
# ---------------------------

def generate_balanced_routing(tokens, top_k_experts, num_experts):
    """
    Randomly assign 'top_k' experts to each token.
    
    Returns a (tokens x top_k) matrix of expert indices.
    """

    return np.random.randint(0, num_experts, size=(tokens, top_k_experts))

def generate_imbalanced_routing(tokens, top_k, num_experts, hot_ratio, hot_weight):
    """
    Generate an imbalanced routing table where some experts are more frequently assigned.
    
    Returns a (tokens x top_k) matrix of expert indices.
    """
    routing = np.zeros((tokens, top_k))
    hot_experts = np.random.randint(0, num_experts, size=int(num_experts * hot_ratio))  # Number of hot experts
    cold_experts = np.setdiff1d(np.arange(num_experts), hot_experts)  # Remaining experts
    for token_id in range(tokens):
        # Assign hot experts with higher probability
        if np.random.rand() < hot_weight:
            routing[token_id] = np.random.choice(hot_experts, size=top_k, replace=False)
        else:
            routing[token_id] = np.random.choice(cold_experts, size=top_k, replace=False)

    return routing
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
    print(routing)
    comm_matrix = np.zeros((num_gpus, num_gpus))  # Communication matrix in bytes
    expert_load = np.zeros(NUM_EXPERTS)  # Track received on each expert
    for token_id, experts in enumerate(routing):
        source_gpu = token_id % num_gpus  # Distribute equal load on each GPU
        for expert in experts:
            target_gpu = expert_gpu_map[expert]
            if source_gpu != target_gpu:
                data_size = token_size * dtype_size
                expert_load[int(expert)] += data_size
                comm_matrix[source_gpu][target_gpu] += data_size

    return comm_matrix, np.round(expert_load / (1024 ** 2), 2)


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
if "__main__" == __name__:
    np.random.seed()  # For reproducibility
    print(f"Simulating communication for {NUM_EXPERTS} experts across {NUM_GPUS} GPUs with {TOKENS} token(s)...")
    print(f"Token Size: {EMBED_DIM}, DType Size: {DTYPE_SIZE} bytes, Top-K: {TOP_K_EXPERTS}")
    print(f"Experts per GPU: {EXPERTS_PER_GPU}")

    if IS_BALANCED:
        print("(Balanced Routing)")
        routing = generate_balanced_routing(TOKENS, TOP_K_EXPERTS, NUM_EXPERTS)
    else:
        print(f"(Imbalanced Routing, Hot Ratio: {HOT_RATIO}, Hot Weight: {HOT_WEIGHT})")
        routing = generate_imbalanced_routing(TOKENS, TOP_K_EXPERTS, NUM_EXPERTS, HOT_RATIO, HOT_WEIGHT)
    expert_gpu_map = get_expert_gpu_map(NUM_EXPERTS, NUM_GPUS)
    comm_matrix, expert_load = simulate_all_to_all(routing, expert_gpu_map, NUM_GPUS, EMBED_DIM, DTYPE_SIZE)
    print(f"Received load per expert : {expert_load}")
    plot_comm_matrix(comm_matrix)
