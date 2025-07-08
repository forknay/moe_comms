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
            node_load[source][expert//NUM_EXPERTS_PER_NODE] += UNIT_COMM_LOAD # Expert weight and ID and chosen expert (irrelevant in the case when there's only one expert per node)
    for node in range(NUM_NODES):
        num_recurrent += node_load[node][node]
        node_load[node][node] = 0

    return node_load, num_recurrent
            
def full_mesh_comm(node_load: dict[int,dict[int,int]] ) -> float:
    comm_time = INITIAL_CPU_DELAY # GPU routing to CPU and initial routing instructions CPU to GPU
    num_rounds = 1
    while True: # Each iteration is one round of data transfer
        if num_rounds > 10:
            return
        dram_map = {i: 0 for i in range(NUM_NODES)} # Each node with the amount of used DRAM
        active_links = {i: {j:[[[0], 0]]*NUM_LINKS for j in range(NUM_NODES)} for i in range(NUM_NODES)} # [[packets], direction] for each link, direction is positive for src->dest, negative for dest->src
        print("ROUND", num_rounds)
        print("NODE LOAD", node_load)
        existing_load = False
        round_time = 0
        for src, dest_loads in node_load.items(): # Assign one round of data
            if sum(dest_loads.values()) == 0: # If no data to send, skip
                continue
            existing_load = True # Check for load for main while loop
            if dram_map[src] + PACKET_SIZE > GPU_DRAM: # If dest node has no space in DRAM, skip
                print(f"Node {src} has no space in DRAM, skipping")
                continue
            transferred = True
            while transferred: # Loop for round robin
                transferred = False
                for dest, load in dest_loads.items():
                    num_packets = 0 # Round robin counter
                    while load > 0 and num_packets < ROUND_ROBIN_MAX_PACKETS:
                        packet_size = min(load, PACKET_SIZE)
                        print(packet_size, dram_map[src])
                        if dram_map[src] + packet_size > GPU_DRAM or dram_map[dest] + packet_size > GPU_DRAM:
                            print(f"Node {src} or {dest} has no more space in DRAM, skipping")
                            break
                        possible_links = [(i, link) for i, link in enumerate(active_links[src][dest]) if ((sum(link[0]) + packet_size) <= INTRA_BW) and (link[1] >= 0)] # Assume reverse link is at the same index
                        print(possible_links)
                        if not possible_links:
                            print(f"No available links for {src} to {dest}, skipping")
                            break
                        else:
                            transferred = True
                        num_packets += 1
                        link = possible_links[0] # Pick the first available link
                        if link[1][1] == 0: # If the link is not used, set it up
                            active_links[src][dest][link[0]] = [[packet_size], 1] # Set direction to src->dest
                            active_links[dest][src][link[0]] = [[packet_size], -1] # Set reverse direction to dest->src
                        else:
                            active_links[src][dest][link[0]][0].append(packet_size)
                            active_links[dest][src][link[0]][0].append(packet_size)

                        dram_map[src] += packet_size
                        dram_map[dest] += packet_size
                        node_load[src][dest] -= packet_size
                        if node_load[src][dest] < 0:
                            raise ValueError(f"Negative load for {src} to {dest}, load: {node_load[src][dest]}")
                        load -= packet_size
                        

        if not existing_load: # All done
            print("No more data to transfer, exiting")
            break
        temp = [i.values() for i in active_links.values()]
        packet_list = []
        for values in temp:
            for value in values:
                packet_list += [value[0][0]]
        largest_packets = max([(sum(i), i) for i in packet_list]) if packet_list else 0
        most_packets = max([(len(i), i) for i in packet_list]) if packet_list else 0
        if largest_packets[1] != most_packets[1]:
            print("WARNING: Largest packet size is not equal to the number of packets")
        print("LARGEST PACKET SIZE", largest_packets[1], "PACKETS", most_packets[1])
        round_time = most_packets[0]*PACKET_PREP_DELAY + largest_packets[0]*INTRA_BW # Convert to time, no parallelization yet
        print(round_time)
        comm_time += BASE_DELAY + round_time # GPU->CPU confirmation of round completion
        num_rounds += 1
        print("ACTIVE LINKS", active_links)
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
            if dma_engines[src][0] < 1 and dma_engines[dest][1] < 1 and (removed.count((src,dest)) + removed.count((dest,src))) < NUM_LINKS:
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
    print("-"*20)
    print(full_mesh_comm(load))
