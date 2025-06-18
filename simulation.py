import numpy as np
import random

NUM_EXPERTS = 256            # Total number of experts in the MoE layer
SEQLEN = 1024                # Number of tokens to simulate
TOP_K = 8                    # Number of routed experts assigned to each token
EMBED_DIM = 7168             # Embedding dimension size

HOT_RATIO = 0.5              # Ratio of hot experts (for imbalanced routing)
HOT_WEIGHT = 0.7             # Weight for hot experts in imbalanced routing

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
        nb_hot_experts = int(NUM_EXPERTS * HOT_RATIO)
        hot_experts = random.sample(range(NUM_EXPERTS), nb_hot_experts)
        cold_experts = list(set(range(NUM_EXPERTS)) - set(hot_experts))

        if nb_hot_experts == NUM_EXPERTS:
            expert_weights = [1 / NUM_EXPERTS] * NUM_EXPERTS
        else:
            expert_weights = (
                [HOT_WEIGHT / nb_hot_experts] * nb_hot_experts +
                [(1 - HOT_WEIGHT) / (NUM_EXPERTS - nb_hot_experts)] * (NUM_EXPERTS - nb_hot_experts)
            )

        all_experts = hot_experts + cold_experts
        routing_table = [
            np.random.choice(all_experts, TOP_K, p=expert_weights, replace=False) for _ in range(SEQLEN)
        ]
        mock_weights = np.random.rand(SEQLEN, TOP_K).astype(np.float32)
        return mock_weights, np.array(routing_table, dtype=np.int8)

if __name__ == "__main__":
    gate_output = Gate.generate_routing()
    print(gate_output[0].shape, gate_output[1].shape)