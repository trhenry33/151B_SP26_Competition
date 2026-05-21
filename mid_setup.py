import subprocess
import sys

packages = [
    "numpy<2",
    "torch==2.5.1",
    "torchvision==0.20.1",
    "torchaudio==2.5.1",
    "sympy",
    "transformers",
    "tqdm",
    "bitsandbytes",
    "antlr4-python3-runtime==4.11.1",
    "ipykernel",
    "jupyter",
    "accelerate",
    "vllm==0.19.1",
]

subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--user",
        "--upgrade",
        "--constraint",
        "constraints.txt",
        *packages,
    ],
    check=True,
)