# moe_comms
MoE All-to-All communications modeling with simulation and performance modeling <br>

**Currently Supports:**
<br>
- Parameters for different MoE configurations and hardware restrictions
- Routing data generation (with load imbalance parameters)
- Routing data conversion to bytes for each node (dst, src)
- Full mesh communication of a cluster (with host) with certain assumptions + Communication time
- Multiple links per connection between nodes (can send fragment of a load that is > intra_bw or send in a different direction)
<br>

**Not supported (yet) / Assumptions:** <br>
- Hierarchical communication (source to cluster, cluster to node)
- Create visualization
- Throughput calculation (prefill / decode)
- Add delay/transfer parallelization for each round (ie each packet needs to prepare but can be prepared while last packet is sending) (list all packets for a round, choose link with most packets, do parallelization for those to find critical path)
- Add comms for allocation ** ask where
- Add flag size for packets
- Currently round robin only applied inside a node load, ie starting at node 0 it will try to finish all node 0 sends before moving to node 1, inefficiency as it might have no receives during first rounds
- PCIe FIFO Buffer Size (smaller packets would have inefficiencies)
-----------------------------------------