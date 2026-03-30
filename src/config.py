from src.utils import utils

# Pallet EUR 1 (mm and kg) - European standard pallet
PALLET_WIDTH = 800  # Standard EUR pallet width
PALLET_LENGTH = 1200  # Standard EUR pallet length
PALLET_HEIGHT = 2000  # Maximum stacking height
PALLET_LOAD = 9780  # Maximum load capacity

# Primary pallet dimensions object used throughout the system
PALLET_DIMS = utils.Dimension(PALLET_WIDTH, PALLET_LENGTH, PALLET_HEIGHT, PALLET_LOAD)

# Product dimension ranges (mm and kg) - Typical consumer goods
MIN_PRODUCT_WIDTH = 125  # Minimum product width
MAX_PRODUCT_WIDTH = 600  # Maximum product width
MIN_PRODUCT_LENGTH = 125  # Minimum product length
MAX_PRODUCT_LENGTH = PALLET_LENGTH  # Maximum product length (pallet-constrained)
MIN_PRODUCT_HEIGHT = 50  # Minimum product height
MAX_PRODUCT_HEIGHT = 400  # Maximum product height
MIN_PRODUCT_WEIGHT = 2  # Minimum product weight
MAX_PRODUCT_WEIGHT = PALLET_LOAD  # Maximum product weight (pallet-constrained)

# Random seed for reproducible experiments
RANDOM_SEED = 42
