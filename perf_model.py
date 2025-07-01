from params import *
from simulation import import_routing
import math
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
    comm_time = INITIAL_CPU_DELAY # GPU routing to CPU
    max_active_links = NUM_LINKS * NUM_NODES * (NUM_NODES - 1) // 2
    print("MAX ACTIVE LINKS:", max_active_links)
    num_rounds = 1
    while True: # Each iteration is one round of data transfer
        dma_engines = {i: [0,0] for i in range(NUM_NODES)} # Each node with the num of send and receive operations (must not exceed NUM_DMA_ENGINES) [send, receive]
        active_links = {i: {j:0 for j in range(NUM_NODES)} for i in range(NUM_NODES)} # Each node has a dict of links and their current load
        print("ROUND", num_rounds)
        print("NODE LOAD", node_load)
        num_active_links = 0 
        existing_load = False
        transfer_delay = 0
        for dest, source in node_load.items(): # Assign one round of data
            if sum(source.values()) == 0:
                continue
            elif dma_engines[dest][1] >= NUM_DMA_ENGINES:
                #print(f"Node {dest} has reached max receive operations (outer)")
                continue
            existing_load = True
            for src, load in source.items():
                if max_active_links == num_active_links:
                    print("MAX ACTIVE LINKS REACHED")
                    break
                elif dma_engines[src][0] >= NUM_DMA_ENGINES:
                    #print(f"Node {src} has reached max send operations (inner)")
                    continue
                while active_links[dest][src] < NUM_LINKS and active_links[src][dest] < NUM_LINKS and load != 0: # Link is free and non-zero load
                    if load < 0:
                        raise ValueError("Negative load detected, check routing and weights")
                    elif dma_engines[src][0] >= NUM_DMA_ENGINES:
                        #print(f"Node {src} has reached max send operations")
                        break
                    elif dma_engines[dest][1] >= NUM_DMA_ENGINES:
                        #print(f"Node {dest} has reached max receive operations")
                        break
                    if min(load, INTRA_BW) > transfer_delay:
                        transfer_delay = min(load, INTRA_BW)
                    dma_engines[src][0] += 1
                    dma_engines[dest][1] += 1
                    active_links[dest][src] += 1
                    active_links[src][dest] += 1
                    
                    node_load[dest][src] -= min(load, INTRA_BW)
                    load -= min(load, INTRA_BW)
                    num_active_links += 1

        if not existing_load: # All done
            if num_active_links != 0:
                raise ValueError("Active links not zero, but no existing load found")
            break
        transfer_delay /= INTRA_BW # Convert to time
        print(transfer_delay)
        comm_time += BASE_DELAY*2 + transfer_delay # One for CPU->GPU  instruction, one for GPU->CPU confirmation of completion/reception
        num_rounds += 1
        print("ACTIVE LINKS", active_links)
        print("NUM ACTIVE LINKS", num_active_links)
        print("DMA ENGINES", dma_engines)
        print("COMM TIME", comm_time)
        print("-"*20)
    return comm_time
def check_rounds(node_load: dict[int,dict[int,int]]) -> int: # Assumes no DMA restrictions
    max_load = set()
    for dest, source in node_load.items():
        for src, load in source.items():
            if src < dest:
                continue
            if node_load[src][dest] > 0:
                if load % INTRA_BW != 0:
                    load += 1
            max_load.add(load + node_load[src][dest])
    return math.ceil(math.ceil(max(max_load) / INTRA_BW) / NUM_LINKS)

def check_rounds_dma(node_load: dict[int,dict[int,int]]) -> int:
    operations = []
    for dest, source in node_load.items():
        for src, load in source.items():
            for _ in range(math.ceil(load / INTRA_BW)):
                operations.append((src,dest))
    num_rounds = 0
    
    while operations:
        removed = []
        dma_engines = {i: [0,0] for i in range(NUM_NODES)} # Each node with the num of send and receive operations (must not exceed NUM_DMA_ENGINES) [send, receive]
        num_rounds += 1
        for src, dest in operations:
            if dma_engines[src][0] < NUM_DMA_ENGINES and dma_engines[dest][1] < NUM_DMA_ENGINES and (removed.count((src,dest)) + removed.count((dest,src))) < NUM_LINKS:
                dma_engines[src][0] += 1
                dma_engines[dest][1] += 1
                removed.append((src, dest))
        for packet in removed:
            operations.remove(packet)
    return num_rounds
        
if __name__ == "__main__":
    weights, routing = import_routing()
    load, num_rec = convert_to_bytes(weights, routing)
    print(f"Load (dest, src) (bytes): {load}, Recurrent load (bytes): {num_rec}")
    print("Total load with recurrent equals expected load for hyperparameters:", sum([sum(i.values()) for i in load.values()])+num_rec == SEQLEN*TOP_K*UNIT_COMM_LOAD)
    print("Expected num of rounds:", check_rounds(load))
    print(f"Expected num of rounds with {NUM_DMA_ENGINES} DMA engines: {check_rounds_dma(load)}")
    print("-"*20)
    print(full_mesh_comm(load))
