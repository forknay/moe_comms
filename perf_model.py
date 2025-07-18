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
            node_load[source][expert//NUM_EXPERTS_PER_NODE] += UNIT_COMM_LOAD # Expert weight and ID and chosen expert (irrelevant in the case when there's only one expert per node)
    for node in range(NUM_NODES):
        num_recurrent += node_load[node][node]
        node_load[node][node] = 0

    return node_load, num_recurrent
            
def full_mesh_comm(node_load: dict[int,dict[int,int]] ) -> float:
    output = ["NODE LOAD", "---------------", str(node_load)]
    comm_time = INITIAL_CPU_DELAY # GPU routing to CPU and initial routing instructions CPU to GPU
    num_rounds = 1
    while True: # Each iteration is one round of data transfer
        output.append("---------------")
        output.append(f"ROUND {num_rounds}")
        nic_map = {i: 0 for i in range(NUM_NODES)} # Each node with the amount of used NIC memory
        active_links = {i: {j:[[[0], 0]]*NUM_LINKS for j in range(NUM_NODES)} for i in range(NUM_NODES)} # [[packets], direction] for each link, direction is positive for src->dest, negative for dest->src
        if DEBUG:
            print("ROUND", num_rounds)
            print("NODE LOAD", node_load)
        existing_load = False
        round_time = 0
        for src, dest_loads in node_load.items(): # Assign one round of data
            if sum(dest_loads.values()) == 0: # If no data to send, skip
                continue
            existing_load = True # Check for load for main while loop
            transferred = True # Initialize
            while transferred: # Loop for round robin
                transferred = False
                for dest, load in dest_loads.items():
                    num_packets = 0 # Round robin counter
                    while load > 0 and num_packets < ROUND_ROBIN_MAX_PACKETS:
                        packet_size = min(load, PACKET_SIZE) # Assuming no flags or other headers
                        if nic_map[src] + packet_size > NIC_RATE:
                            print(f"Node {src} has no more space in DRAM, skipping") if DEBUG else None
                            #output.append(f"Node {src} has no more space in DRAM, skipping")
                            break
                        elif nic_map[dest] + packet_size > NIC_RATE:
                            print(f"Node {dest} has no more space in DRAM, skipping") if DEBUG else None
                            #output.append(f"Node {dest} has no more space in DRAM, skipping")
                            break
                        # Find all links where the sent packets do not exceed the intra-node bandwidth and the link is in the correct direction (or not used)
                        possible_links = sorted([(i, link) for i, link in enumerate(active_links[src][dest]) if ((sum(link[0]) + packet_size) <= INTRA_BW) and (link[1] >= 0)]) # Assume reverse link is at the same index
                        # Currently take first available link (implies already used links first if they are not full). Could be edited to take non-used links first for parallelization rather than maximum utilization of each link
                        if not possible_links:
                            print(f"No available links for {src} to {dest}, skipping") if DEBUG else None
                            #output.append(f"No available links for {src} to {dest}, skipping")
                            break

                        transferred = True
                        num_packets += 1
                        link = possible_links[0] # Pick the first available link
                        if link[1][1] == 0: # If the link is not used, set it up
                            active_links[src][dest][link[0]] = [[packet_size], 1] # Set direction to src->dest
                            active_links[dest][src][link[0]] = [[packet_size], -1] # Set reverse direction to dest->src
                        else: # Add packet to existing link
                            active_links[src][dest][link[0]][0].append(packet_size)
                            active_links[dest][src][link[0]][0].append(packet_size)

                        nic_map[src] += packet_size
                        nic_map[dest] += packet_size
                        node_load[src][dest] -= packet_size
                        if node_load[src][dest] < 0:
                            raise ValueError(f"Negative load for {src} to {dest}, load: {node_load[src][dest]}")
                        load -= packet_size

        if not existing_load: # All done
            print("No more data to transfer, exiting") if DEBUG else None
            output.append("No more data to transfer, exiting")
            break
        # Find link with most transferred packets & largest cumulative size (should usually be the same unless bunch of small packets are sent)
        temp = [i.values() for i in active_links.values()]
        packet_list = []
        for values in temp:
            for value in values:
                packet_list += [value[0][0]]
        largest_packets = max([(sum(i), i) for i in packet_list]) if packet_list else 0
        most_packets = max([(len(i), i) for i in packet_list]) if packet_list else 0
        if largest_packets[1] != most_packets[1]:
            print("WARNING: Largest packet size is not equal to the largest number of packets")
        round_time = most_packets[0]*PACKET_PREP_DELAY + largest_packets[0]/INTRA_BW # Convert to time, no parallelization yet
        comm_time += BASE_DELAY + round_time # GPU->CPU confirmation of round completion
        num_rounds += 1
        if DEBUG:
            print("LARGEST PACKET SIZE", largest_packets, "MOST PACKETS", most_packets)
            print("Round time:", round_time)
            print("ACTIVE LINKS", active_links)
            print("COMM TIME", comm_time)
            print("-"*20)
        output.append(f"LARGEST PACKET SIZE: {largest_packets}, MOST PACKETS: {most_packets}")
        output.append(f"Round time: {round_time} ms")
        output.append(f"ACTIVE LINKS: {active_links}")
        output.append(f"NIC MAP: {nic_map}")
        output.append(f"COMM TIME: {comm_time} ms")
    file = open("comm_log.txt", "w")
    file.write("\n".join(output))
    file.close()
    return comm_time

def check_rounds(node_load: dict[int,dict[int,int]]) -> int: # Needs implementation
    pass

if __name__ == "__main__":
    weights, routing = import_routing()
    load, num_rec = convert_to_bytes(weights, routing)
    print(f"Load (src, dest) (bytes): {load}, Recurrent load (bytes): {num_rec}")
    print("-"*20)
    print("Total load with recurrent equals expected load for hyperparameters:", sum([sum(i.values()) for i in load.values()])+num_rec == SEQLEN*TOP_K*UNIT_COMM_LOAD)
    print("Comm time:", full_mesh_comm(load), "ms")