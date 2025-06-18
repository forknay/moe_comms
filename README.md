# moe_comms
MoE All-to-All communications modeling with simulation and performance modeling

Added :

-Basic Parametrization for MoE & Hardware Restrictions

-Heatmap visualization of congestion patterns

Dataset Generator (Initializing EP & Routing matrices)
    Parametrize to generate custom datasets

----------------------------------------------------------------------

TO ADD (PERFORMANCE MODEL): **assuming the use of theoretical numbers & no empirical tests (i have no gpus lol)

-Expert load imbalance
-Compute load/expert

Latency Modeling
    Average/Worst Case latency
    Bandwidth Restrictions
    Sync time for AlltoAll 
    Network Topology constraints

Hierarchical communication (source to cluster, cluster to node)
    Different Bandwidth depending on level

Add communication/compute overlap

Create visualization

Throughput calculation (prefill / decode)

-----------------------------------------

(Potentially add DP + AllReduce time analysis)