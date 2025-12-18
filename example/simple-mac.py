from dia.model import Dia


# Use float32 for better stability on MPS (Apple Silicon)
# float16 may have precision issues on MPS and provides minimal performance benefit
model = Dia.from_pretrained("nari-labs/Dia-1.6B-0626", compute_dtype="float32")

text = "[S1] Dia is an open weights text to dialogue model. [S2] You get full control over scripts and voices. [S1] Wow. Amazing. (laughs) [S2] Try it now on Git hub or Hugging Face."

# It is important to set the `use_torch_compile` argument to `False` when using Dia on MacOS.
# This is because the `torch.compile` function is not supported on MacOS.
output = model.generate(text, use_torch_compile=False, verbose=True)

model.save_audio("simple.mp3", output)
