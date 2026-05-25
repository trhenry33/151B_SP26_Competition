import subprocess
import sys

def run(cmd):
    print("\nRunning:", " ".join(cmd), "\n")
    subprocess.run(cmd, check=True)

PY = sys.executable

# Core packages first, without pinning torch.
# vllm==0.19.1 wants torch==2.10.0, so do NOT force torch==2.5.1.
run([
    PY, "-m", "pip", "install", "--upgrade",
    "numpy<2",
    "sympy",
    "transformers",
    "tqdm",
    "bitsandbytes",
    "antlr4-python3-runtime==4.11.1",
    "ipykernel",
    "jupyter",
    "accelerate",
    "peft",
    "trl",
    "datasets",
])

# Install vLLM separately so it can pull the Torch version it needs.
run([
    PY, "-m", "pip", "install", "--upgrade",
    "vllm==0.19.1",
])

# Force NumPy back below 2 because scipy/sklearn/tensorflow/pyarrow may break otherwise.
run([
    PY, "-m", "pip", "install", "--force-reinstall",
    "numpy<2",
])

print("\n=== VERIFYING INSTALLS ===\n")

checks = [
    'import numpy; print("numpy:", numpy.__version__)',
    'import torch; print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())',
    'import vllm; print("vllm:", vllm.__version__)',
    'import transformers; print("transformers:", transformers.__version__)',
    'import peft, trl, datasets; print("peft/trl/datasets: OK")',
]

for code in checks:
    run([PY, "-c", code])

print("\nSetup complete.")