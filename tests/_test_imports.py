import sys
sys.path.insert(0, 'F:/wkspace/fitness-video-pipeline')
print("Testing imports...")
import argparse
print("argparse ok")
import cv2
print("cv2 ok")
import numpy as np
print("numpy ok")
from pathlib import Path
print("Path ok")

# Test problematic imports
try:
    import importlib
    io = importlib.import_module('stages.20_intro_outro')
    print("20_intro_outro ok:", io)
except Exception as e:
    print(f"20_intro_outro failed: {e}")
    import traceback
    traceback.print_exc()

# Test main.py argparse
try:
    import main
    print("main.py ok")
    # Now try to build parser
    parser = main.build_single_parser()
    print("build_single_parser ok:", parser)
except Exception as e:
    print(f"main.py failed: {e}")
    import traceback
    traceback.print_exc()