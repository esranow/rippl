
import sys
import os

# Add repo root to path
repo_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, repo_root)

print(f"Added {repo_root} to sys.path")

try:
    import ripple
    print(f"SUCCESS: Imported ripple from {ripple.__file__}")
    
    from ripple.physics_blocks import HybridLaplacianBlock
    print("SUCCESS: Imported HybridLaplacianBlock")
    
    block = HybridLaplacianBlock(mode="point", correction_hidden=16)
    print("SUCCESS: Instantiated HybridLaplacianBlock")
    
except ImportError as e:
    print(f"FAILURE: {e}")
except Exception as e:
    print(f"FAILURE: {e}")

