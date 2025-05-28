# moe_comms
MoE All-to-All communications modeling with simulation and performance modeling

Added :

-Basic Parametrization for MoE & Hardware Restrictions
-Expert load imbalance
-Heatmap visualization of congestion patterns
-Compute load/expert

-----------------------------

TO ADD (SIMULATION):

Dataset Generator (Initializing EP & Routing matrices)
    Parametrize to generate custom datasets
        ? Restrain max amount of tokens (don't want too many dropped)

----------------------------------------------------------------------

TO ADD (PERFORMANCE MODEL): **assuming the use of theoretical numbers & no empirical tests (i have no gpus lol)

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