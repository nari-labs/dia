import sys
from random import choice

import torch

from dia.model import Dia


# Only configure Triton on platforms where it's available (Linux/Windows)
if sys.platform in ("linux", "win32"):
    torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.triton.unique_kernel_names = True
    torch._inductor.config.fx_graph_cache = True

# debugging
torch._logging.set_logs(graph_breaks=True, recompiles=True)

model_name = "nari-labs/Dia-1.6B-0626"
compute_dtype = "float16"

model = Dia.from_pretrained(model_name, compute_dtype=compute_dtype)


test_cases = [
    "[S1] Dia is an open weights text to dialogue model.",
    "[S1] Dia is an open weights text to dialogue model. [S2] You get full control over scripts and voices. [S1] Wow. Amazing. (laughs) [S2] Try it now on Git hub or Hugging Face.",
    "[S1] torch.compile is a new feature in PyTorch that allows you to compile your model with a single line of code.",
    "[S1] torch.compile is a new feature in PyTorch that allows you to compile your model with a single line of code. [S2] It is a new feature in PyTorch that allows you to compile your model with a single line of code.",
]


# Wram up
for _ in range(2):
    text = choice(test_cases)
    output = model.generate(text, audio_prompt="./example_prompt.mp3", use_torch_compile=True, verbose=True)
    output = model.generate(text, use_torch_compile=True, verbose=True)

# Benchmark
for _ in range(10):
    text = choice(test_cases)
    output = model.generate(text, use_torch_compile=True, verbose=True)
    output = model.generate(text, audio_prompt="./example_prompt.mp3", use_torch_compile=True, verbose=True)
