import sys
import traceback

try:
    import main
    print("SUCCESS")
except Exception as e:
    print(f"FAILED: {e}")
    traceback.print_exc()
