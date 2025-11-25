
import os
import sys
from dotenv import load_dotenv

# Load env vars
load_dotenv()

print(f"Python executable: {sys.executable}")
print(f"Current working directory: {os.getcwd()}")

try:
    import deepseek
    print("✓ deepseek module imported")
except ImportError as e:
    print(f"❌ Failed to import deepseek: {e}")
    sys.exit(1)

try:
    from deepseek.client import DeepSeekBrain
    print("✓ DeepSeekBrain imported")
except ImportError as e:
    print(f"❌ Failed to import DeepSeekBrain: {e}")
    sys.exit(1)

try:
    from core.system_context import SystemContext
    context = SystemContext()
    print("✓ SystemContext initialized")
    
    client = DeepSeekBrain(context, None)
    print("✓ DeepSeekBrain initialized")
except Exception as e:
    print(f"❌ Failed to initialize DeepSeekBrain: {e}")
    sys.exit(1)
