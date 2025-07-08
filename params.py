# Config parameters for the simulation

TEST_PARAMS = True

if TEST_PARAMS:
    NUM_LAYERS = 1
    NUM_EXPERTS = 4             # Total number of experts in the MoE layer
    SEQLEN = 5                  # Number of tokens to simulate
    TOP_K = 2                    # Number of routed experts assigned to each token
    EMBED_DIM = 16               # Embedding dimension size

    HOT_RATIO = 0.5              # Ratio of hot experts 
    HOT_WEIGHT = 0.8             # Weight for hot experts 
    NUM_HOT_EXPERTS = int(NUM_EXPERTS * HOT_RATIO)  # Number of hot experts

    NUM_NODES = 4

    NUM_EXPERTS_PER_NODE = NUM_EXPERTS // NUM_NODES
    UNIT_COMM_LOAD = 4

    # Infrastructure 
    NUM_LINKS = 1 # Number of links between two nodes
    NUM_DMA_ENGINES = 1 # Number of full-duplex DMA engines per node (determines how parallel the communication can be), no implementation yet, assume infinite engines
    BASE_DELAY = 2 # in ms
    INITIAL_CPU_DELAY = 0 # in ms, delay for GPU to send routing to CPU
    INTRA_BW = 2 # in B/ms just using intra for now, no implementation for different clusters just yet
    INTER_BW = 1 # in B/ms
    PACKET_SIZE = 1 # in bytes
    PACKET_PREP_DELAY = 1 # in ms
    PARALLELIZATION_MULTIPLIER = 1.2 # Extra time needed if done in parallel
    ROUND_ROBIN_MAX_PACKETS = 2 # Max packets before switching to another node (could come back if no other nodes have packets to send)
    GPU_DRAM = 10

else:
    NUM_LAYERS = 61
    NUM_EXPERTS = 128            # Total number of experts in the MoE layer
    SEQLEN = 1024                # Number of tokens to simulate
    TOP_K = 8                    # Number of routed experts assigned to each token
    EMBED_DIM = 7168             # Embedding dimension size

    HOT_RATIO = 0.5              # Ratio of hot experts 
    HOT_WEIGHT = 0.8             # Weight for hot experts 
    NUM_HOT_EXPERTS = int(NUM_EXPERTS * HOT_RATIO)  # Number of hot experts

    NUM_NODES = 8

    # Data conversion
    WEIGHT_PRECISION = 4 # FP32 bytes
    ROUTING_PRECISION = 1 # INT8 bytes
    ID_PRECISION = 2 # INT16 bytes
    NUM_EXPERTS_PER_NODE = NUM_EXPERTS // NUM_NODES
    UNIT_COMM_LOAD = (WEIGHT_PRECISION + 2*ID_PRECISION + ROUTING_PRECISION)

    # Infrastructure 
    NUM_LINKS = 1 # Number of links between two nodes
    NUM_DMA_ENGINES = 1 # Number of full-duplex DMA engines per node (determines how parallel the communication can be), no implementation yet, assume infinite engines
    BASE_DELAY = 2 # in ms
    INTRA_BW = 100 # in B/ms just using intra for now, no implementation for different clusters just yet
    INTER_BW = 50 # in B/ms
    PACKET_SIZE = 128 # in bytes
    PACKET_PREP_DELAY = 1 # in ms
    PARALLELIZATION_MULTIPLIER = 1.2 # Extra time needed if done in parallel
    ROUND_ROBIN_MAX_PACKETS = 2 # Max packets before switching to another node (could come back if no other nodes have packets to send)
    GPU_DRAM = 1000


assert NUM_EXPERTS % NUM_NODES == 0, "Number of experts must be divisible by number of nodes"
assert NUM_LINKS >= 1, "Number of links must be at least 1"
assert INTRA_BW > 0 and INTER_BW > 0, "Bandwidth must be greater than 0"
assert UNIT_COMM_LOAD > 0, "Unit communication load must be greater than 0"