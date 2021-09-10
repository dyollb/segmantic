import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# add methods in core visible at top-level
from core import extract_slices, scale_to_range, resample, pad, crop

# CT, ... specific methods can be used from modality
import modality