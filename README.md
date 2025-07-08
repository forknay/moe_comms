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
- Packet loss and retransmission
- Memory constraints (would need extra comms for allocation) *
- Hierarchical communication (source to cluster, cluster to node)
- Create visualization
- Throughput calculation (prefill / decode)
-  Fix packet size being fixed issue ****
- could still be dest, src in simulation, will have to make sure (should be fine, i dont think dest src is identified there)
- Multiple Layers == Independent
- Round robin data packet sending with limit per packet
- Add packet size, should be separate from bandwidth consideration
- Add delay/transfer parallelization for each round (ie each packet needs to prepare but can be prepared while last packet is sending) (list all packets for a round, choose link with most packets, do parallelization for those to find critical path)
- Add comms for allocation
- Separate packets that may overlap over two links 
-----------------------------------------