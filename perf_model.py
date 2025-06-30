from params import *
from simulation import import_routing
## ASSUMING FULL MESH TOPOLOGY with cpu to send directions to DMA engines

# Assumptions
# No loss
# No overhead
# No memory allocation constraints, could just reserve space for decoding, not sure for prefill

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
            
def full_mesh_comm(node_load: dict[int,dict[int,int]] ) -> float:
    comm_time = BASE_DELAY # GPU routing to CPU
    max_active_links = NUM_LINKS * NUM_NODES * (NUM_NODES - 1) // 2
    num_rounds = 0
    while True: # Each iteration is one round of data transfer
        active_links = {i: {j:0 for j in range(NUM_NODES)} for i in range(NUM_NODES)} # Each node has a dict of links and their current load
        num_rounds += 1
        print("ROUND", num_rounds)
        print("NODE LOAD", node_load)
        num_active_links = 0 
        existing_instruction = False
        existing_load = False
        transfer_delay = 0
        for dest, source in node_load.items(): # Assign one round of data
            if sum(source.values()) == 0:
                continue
            existing_load = True
            for src, load in source.items():
                if max_active_links == num_active_links:
                    print("MAX ACTIVE LINKS REACHED")
                    break
                while active_links[dest][src] < NUM_LINKS and active_links[src][dest] < NUM_LINKS and load != 0: # Link is free and non-zero load
                    if load < 0:
                        raise ValueError("Negative load detected, check routing and weights")
                    if min(load, INTRA_BW) > transfer_delay:
                        transfer_delay = min(load, INTRA_BW)
                    existing_instruction = True
                    active_links[dest][src] += 1
                    active_links[src][dest] += 1
                    node_load[dest][src] -= min(load, INTRA_BW)
                    load -= min(load, INTRA_BW)
                    num_active_links += 1
        if existing_load and not existing_instruction:
            print(active_links)
            raise ValueError("No existing instruction, but existing load found, this is unexpected")
        transfer_delay /= INTRA_BW # Convert to time
        if existing_instruction:
            comm_time += BASE_DELAY*2 + transfer_delay # One for CPU->GPU  instruction, one for GPU->CPU confirmation of completion/reception
            print("comms")
        if not existing_load: # All done
            if num_active_links != 0:
                raise ValueError("Active links not zero, but no existing load found")
            break
        print("ACTIVE LINKS", active_links)
        print("NUM ACTIVE LINKS", num_active_links)
        
        print(comm_time)
    return comm_time

if __name__ == "__main__":
    weights, routing = import_routing()
    load, num_rec = convert_to_bytes(weights, routing)
    print(load, num_rec)
    print(sum([sum(i.values()) for i in load.values()])+num_rec == SEQLEN*TOP_K*UNIT_COMM_LOAD)
    print(full_mesh_comm(load))
