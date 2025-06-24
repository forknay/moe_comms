from simulation import *

## ASSUMING FULL MESH TOPOLOGY 

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
HOST = True
BASE_DELAY = 2 # in ms
INTRA_BW = 100 # in GB/s
INTER_BW = 50 # in GB/s

def convert_to_bytes(weights, routing):
    d = {i: 0 for i in range(NUM_NODES)}
    nb_recurrent = 0
    for i, route in enumerate(routing):
        route = route.tolist()
        
        if int(weights[i][-2]) in route: # Remove one routing if already on the right NPU
            route.remove(int(weights[i][-2]))
            nb_recurrent += 1
        for expert in route:
            d[expert//NUM_EXPERTS_PER_NODE] += UNIT_COMM_LOAD # Expert weight and ID and chosen expert (irrelevant in the case when there's only one expert per node)
    return d, nb_recurrent
            

if __name__ == "__main__":
    weights, routing = import_routing()
    load, nb_rec = convert_to_bytes(weights, routing)
    print(load, nb_rec)