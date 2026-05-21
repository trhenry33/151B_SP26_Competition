import subprocess
import sys

commands = [
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--user",
        "--upgrade",
        "numpy<2",
        "sympy",
        "transformers",
        "tqdm",
        "bitsandbytes",
        "antlr4-python3-runtime==4.11.1",
        "ipykernel",
        "jupyter",
        "accelerate",
    ],
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--user",
        "--upgrade",
        "vllm==0.19.1",
    ],
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--user",
        "--force-reinstall",
        "numpy<2",
    ],
]

for cmd in commands:
    print("\nRunning:", " ".join(cmd), "\n")
    subprocess.run(cmd, check=True)

print("\n=== VERIFYING INSTALLS ===\n")

verify_commands = [
    'import numpy; print("numpy:", numpy.__version__)',
    'import torch; print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())',
    'import vllm; print("vllm:", vllm.__version__)',
]

for code in verify_commands:
    subprocess.run([sys.executable, "-c", code], check=True)

print("\nSetup complete.")