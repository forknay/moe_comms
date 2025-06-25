from simulation import *

## ASSUMING FULL MESH TOPOLOGY with cpu to send directions to DMA engines

# Assumptions
# No loss
# No overhead
# No memory allocation constraints, could just reserve space for decoding, not sure for prefill

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
INTER_BW = 50e6 # in B/ms

def convert_to_bytes(weights, routing):
    node_load = {i: {j:0 for j in range(NUM_NODES)} for i in range(NUM_NODES)}
    num_recurrent = 0
    for i, route in enumerate(routing):
        source = int(weights[i][-2])
        route = route.tolist()
        for expert in route:
            node_load[expert//NUM_EXPERTS_PER_NODE][source] += UNIT_COMM_LOAD # Expert weight and ID and chosen expert (irrelevant in the case when there's only one expert per node)
    for node in range(NUM_NODES):
        num_recurrent += node_load[node][node]
        node_load[node][node] = 0

    return node_load, num_recurrent
            
def full_mesh_comm(node_load):
    comm_time = 0
    num_active_links = 0 
    max_active_links = NUM_LINKS * NUM_NODES * (NUM_NODES - 1) // 2
    active_links = {i: {j:0 for j in range(NUM_NODES)} for i in range(NUM_NODES)} # Each node has a dict of links and their current load
    while True:
        print("NODE LOAD", node_load)
        existing_instruction = False
        existing_load = False
        for dest, source in node_load.items(): # Assign one round of data
            if sum(source.values()) == 0:
                continue
            existing_load = True
            for src, load in source.items():
                if max_active_links == num_active_links:
                    break
                elif active_links[dest][src] == 0 and active_links[src][dest] == 0 and load > 0: # Link is free and non-zero load
                    existing_instruction = True
                    active_links[dest][src] = load
                    node_load[dest][src] = 0
                    num_active_links += 1
        if existing_instruction:
            comm_time += BASE_DELAY*2 # One for CPU instruction, one for one for confirmation of completion
            print("comms")
        if not existing_load: # All done
            break
        print("ACTIVE LINKS", active_links)
        for dest, source in active_links.items(): # Transmit one round of data
            round_time = [0]
            for src, load in source.items():
                if load == 0:
                    continue
                transferred_load = min(load, INTRA_BW)
                if transferred_load == load: # Finished transmitting
                    num_active_links -= 1
                    
                round_time.append(transferred_load/INTRA_BW)
                active_links[dest][src] -= transferred_load
            comm_time += max(round_time)
    return comm_time

if __name__ == "__main__":
    weights, routing = import_routing()
    load, num_rec = convert_to_bytes(weights, routing)
    #print(load, num_rec)
    print(sum([sum(i.values()) for i in load.values()])+num_rec == SEQLEN*TOP_K*UNIT_COMM_LOAD)
    print(full_mesh_comm(load))