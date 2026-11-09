"""GNN surrogate for molecular docking scores with uncertainty-guided active learning.

This package measures how well a graph neural network imitates a physics-based
docking oracle, and how much that surrogate reduces the number of oracle calls
needed to find top-scoring compounds. It makes no claim about experimental
binding affinity; see the README's scientific framing.
"""

__version__ = "0.1.0"
