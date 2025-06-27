# Config parameters for the simulation

TEST_PARAMS = True

if TEST_PARAMS:
    NUM_LAYERS = 1
    NUM_EXPERTS = 10             # Total number of experts in the MoE layer
    SEQLEN = 128                  # Number of tokens to simulate
    TOP_K = 4                    # Number of routed experts assigned to each token
    EMBED_DIM = 16               # Embedding dimension size

    HOT_RATIO = 0.5              # Ratio of hot experts 
    HOT_WEIGHT = 0.8             # Weight for hot experts 
    NUM_HOT_EXPERTS = int(NUM_EXPERTS * HOT_RATIO)  # Number of hot experts

    NUM_NODES = 10

    NUM_EXPERTS_PER_NODE = NUM_EXPERTS // NUM_NODES
    UNIT_COMM_LOAD = 1

    # Infrastructure 
    NUM_LINKS = 1 # Number of links between two nodes
    NUM_DMA_ENGINES = 1 # Number of full-duplex DMA engines per node (determines how parallel the communication can be), no implementation yet, assume infinite engines
    BASE_DELAY = 2 # in ms
    INTRA_BW = 2 # in B/ms just using intra for now, no implementation for different clusters just yet
    INTER_BW = 1 # in B/ms
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
    INTRA_BW = 100e6 # in B/ms just using intra for now, no implementation for different clusters just yet
    INTER_BW = 50e6 # in B/ms


assert NUM_EXPERTS % NUM_NODES == 0, "Number of experts must be divisible by number of nodes"
assert NUM_LINKS >= 1, "Number of links must be at least 1"
assert INTRA_BW > 0 and INTER_BW > 0, "Bandwidth must be greater than 0"
assert UNIT_COMM_LOAD > 0, "Unit communication load must be greater than 0"