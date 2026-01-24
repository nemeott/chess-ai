import numpy as np
import numba as nb
from numba import jit
from timeit import default_timer

@jit(parallel=True)
def _recursive_function(nodes, depth):
    nodes += 1

    if depth == 0:
        return
    
    for i in nb.prange(35):
        _recursive_function(nodes, depth - 1)

class RecursiveFunction:
    def __init__(self):
        self.nodes = 0

    def recursive_function(self, depth):
        _recursive_function((self.nodes), depth)

rec = RecursiveFunction()

# Time the function
start = default_timer()
rec.recursive_function(5)
end = default_timer()
print("Number of nodes:", rec.nodes)
print("Time taken:", end - start)
