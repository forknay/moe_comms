# moe_comms
MoE All-to-All communications modeling with simulation and performance modeling

-----------------------------

TO ADD (SIMULATION):

Expert load imbalance

Compute Bandwidth Requirement
See congestion patterns
Compute load/expert

Dataset Generator (Initializing EP & Routing matrices)
    Parametrize to generate custom datasets
        Network Topology constraints
        Different load distributions (ie: how much entropy)
        Restrain max amount of tokens (don't want too many dropped)

----------------------------------------------------------------------

TO ADD (PERFORMANCE MODEL):

Latency Modeling
    Average/Worst Case latency
    Bandwidth Restrictions
    Hardware Restrictions (GPUs)
    Sync time for AlltoAll 

Hierarchical communication (source to cluster, cluster to node)
    Different Bandwidth depending on level

Add communication/compute overlap

Create visualization

Throughput calculation (prefill / decode)

-----------------------------------------

(Potentially add DP + AllReduce time analysis)