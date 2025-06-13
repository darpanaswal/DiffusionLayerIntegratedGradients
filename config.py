
"""
Configuration settings for DLIG attribution analysis.
"""

# Model configuration
MODEL_PATH = "./models/Dream"
LAYER_NAME = 'model.embed_tokens'  # or any Transformer layer like 'model.layers.10'

# DLIG computation parameters
INTEGRATION_STEPS = 20  # Number of integration steps (m in the formula)

# Generation parameters
MAX_NEW_TOKENS = 64
GENERATION_STEPS = 5

# Output configuration
OUTPUT_DIR = "./attribution_csv"
CSV_FILENAME = "dlig_attribution_scores.csv"

# Device configuration (can be overridden)
DEVICE_MAP = "auto"
TORCH_DTYPE = "bfloat16"
