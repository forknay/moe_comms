from simulation import *

## ASSUMING FULL MESH TOPOLOGY with cpu to send directions to DMA engines

# Assumptions
# No loss
# No overhead
# No memory allocation constraints, could just reserve space for decoding, not sure for prefill

NUM_LAYERS = 61
# Data conversion
WEIGHT_PRECISION = 4 # FP32 bytes
ROUTING_PRECISION = 1 # INT8 bytes
ID_PRECISION = 2 # INT16 bytes
NUM_EXPERTS_PER_NODE = NUM_EXPERTS // NUM_NODES
UNIT_COMM_LOAD = (WEIGHT_PRECISION + 2*ID_PRECISION + ROUTING_PRECISION)

# Infrastructure 
NUM_LINKS = 1 # Number of links between two nodes
NUM_DMA_ENGINES = 1 # Number of full-duplex DMA engines per node (determines how parallel the communication can be)
BASE_DELAY = 2 # in ms
INTRA_BW = 100 # in GB/s
INTER_BW = 50 # in GB/s

def convert_to_bytes(weights, routing):
    node_load = {i: {j:0 for j in range(NUM_NODES)} for i in range(NUM_NODES)}
    nb_recurrent = 0
    for i, route in enumerate(routing):
        source = int(weights[i][-2])
        route = route.tolist()
        
        if source in route: # Remove one routing if already on the right NPU
            route.remove(source)
            nb_recurrent += 1
        for expert in route:
            node_load[expert//NUM_EXPERTS_PER_NODE][source] += UNIT_COMM_LOAD # Expert weight and ID and chosen expert (irrelevant in the case when there's only one expert per node)
    return node_load, nb_recurrent
            
def full_mesh_comm(node_load):
    done = False
    comm_time = 0
    active_links = {i:[0]*NUM_NODES for i in range(NUM_NODES)} # Each node has a list of links and their status (index of the list is the destination node)
    while not done:
        pass

if __name__ == "__main__":
    weights, routing = import_routing()
    load, nb_rec = convert_to_bytes(weights, routing)
    print(load, nb_rec)