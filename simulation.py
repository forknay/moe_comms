import numpy as np
import random

NUM_EXPERTS = 128            # Total number of experts in the MoE layer
SEQLEN = 1024                # Number of tokens to simulate
TOP_K = 8                    # Number of routed experts assigned to each token
EMBED_DIM = 7168             # Embedding dimension size

HOT_RATIO = 0.5              # Ratio of hot experts (for imbalanced routing)
HOT_WEIGHT = 0.8             # Weight for hot experts in imbalanced routing
NB_HOT_EXPERTS = int(NUM_EXPERTS * HOT_RATIO)  # Number of hot experts

NB_NODES = 8

class Gate:
    """Mixture-of-Experts routing gate simulation."""

    @staticmethod
    def generate_routing() -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate routing of tokens to experts.

        Returns:
            mock_weights (np.ndarray): Random weights for each token-expert pair.
            routing_table (np.ndarray): Routing table of expert indices per token.
        """
        hot_experts = random.sample(range(NUM_EXPERTS), NB_HOT_EXPERTS)
        #print(hot_experts)
        cold_experts = list(set(range(NUM_EXPERTS)) - set(hot_experts))
        if NB_HOT_EXPERTS == NUM_EXPERTS:
            expert_weights = [1 / NUM_EXPERTS] * NUM_EXPERTS
        else:
            # Assuming we want a balanced distribution within groups
            expert_weights = ( 
                [HOT_WEIGHT / NB_HOT_EXPERTS] * NB_HOT_EXPERTS +
                [(1 - HOT_WEIGHT) / (NUM_EXPERTS - NB_HOT_EXPERTS)] * (NUM_EXPERTS - NB_HOT_EXPERTS)
            )

        all_experts = hot_experts + cold_experts
        routing_table = [
            np.random.choice(all_experts, TOP_K, p=expert_weights, replace=False) for _ in range(SEQLEN)
        ]
        # Change data types as desired, note that int8 only goes up to 127, so int16 is needed for nb_experts > 128
        mock_weights = np.random.rand(SEQLEN, TOP_K).astype(np.float32)
        return mock_weights, np.array(routing_table, dtype=np.int8)
    
    @staticmethod
    def npu_identify(routing: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """
        Identify the source NPU for each token and batch
        Returns:
            routing (tuple[np.ndarray, np.ndarray]): Updated token weights with NPU indices for combine and token indices for reordering (batch can be determined by token index).
        """
        mock_weights, routing_table = routing
        mock_weights = [(i % NB_NODES, i, mock_weights[i]) for i in range(SEQLEN)]
        return mock_weights, routing_table
    
if __name__ == "__main__":
    gate_output = Gate.generate_routing()
    print("Shapes: ", gate_output[0].shape, gate_output[1].shape)
    print("Samples: ", gate_output[0][0], gate_output[1][0])
    labelled_output = Gate.npu_identify(gate_output)
    print("Labelled output (NPU, token, weight): ", labelled_output[0][0], labelled_output[1][0])
    true_hot_experts = np.argpartition(np.unique(labelled_output[1], return_counts=True)[1], -NB_HOT_EXPERTS)[-NB_HOT_EXPERTS:]
    #print("Check hot_experts: ", true_hot_experts)
    print("Hot experts load: ", sum(np.unique(labelled_output[1], return_counts=True)[1][true_hot_experts])/(SEQLEN*TOP_K))