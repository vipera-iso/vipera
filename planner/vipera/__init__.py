"""vipera — an optimization layer for local LLM model files.

Stage 1: read a GGUF, inventory it, and plan weight placement for a weak
hardware target. Stage 2: model a hybrid-MoE architecture per layer and
schedule how its weights stream across VRAM/RAM/disk. Nothing here loads tensor
data, so it is cheap and safe even for a 300 GiB model.
"""

__version__ = "0.0.1"
