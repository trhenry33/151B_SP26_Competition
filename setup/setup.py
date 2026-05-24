# import sys
# !{sys.executable} -m pip install --user "vllm==0.19.1"
# !pip install --user --force-reinstall "numpy<2"


import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "--user", "vllm==0.19.1"])
subprocess.run([sys.executable, "-m", "pip", "install", "--user", "--force-reinstall", "numpy<2"])