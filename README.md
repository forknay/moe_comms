# moe_comms
MoE All-to-All communications modeling with simulation and performance modeling <br>

** Currently supports: **<br>
- Parameters for different MoE configurations and hardware restrictions
- Routing data generation (with load imbalance parameters)
- Routing data conversion to bytes for each node (dst, src)
- Full mesh communication of a cluster (with host) with certain assumptions + Communication time

<br>

** Not supported (yet) / Assumptions: ** <br>
- Packet loss and retransmission
- Memory constraints (would need extra comms for allocation)
- Prefill/Decode stage separation
- Hierarchical communication (source to cluster, cluster to node)
- Create visualization
- Throughput calculation (prefill / decode)
- Multiple links
- Multiple DMA_engines
- Multiple layers

-----------------------------------------

(Potentially add DP + AllReduce time analysis)