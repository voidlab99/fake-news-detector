import sys
from pathlib import Path

# Make the project root importable (data_utils, app) when running `pytest` from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
