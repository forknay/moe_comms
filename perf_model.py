from simulation import *

WEIGHT_PRECISION = 4 # FP32 bytes
ROUTING_PRECISION = 1 # INT8 bytes
def convert_to_bytes(weights, routing):
    d = {i: 0 for i in range(NB_NODES)}