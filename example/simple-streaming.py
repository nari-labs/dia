import numpy as np

from dia.model import Dia


model = Dia.from_pretrained("nari-labs/Dia-1.6B-0626", compute_dtype="float16")

print("--- Running Test Case 1: Standard Streaming (Not Compiled) ---")

text = "[S1] Dia is an open weights text to dialogue model. [S2] You get full control over scripts and voices. [S1] Wow. Amazing. (laughs) [S2] Try it now on Git hub or Hugging Face."

full_audio = []
for chunk in model.generate_streaming(
    text,
    use_torch_compile=False,
    verbose=True,
    chunk_size=256,
):
    print(f"Received audio chunk of shape: {chunk.shape}")
    full_audio.append(chunk)

if full_audio:
    final_audio_np = np.concatenate(full_audio, axis=0)
    model.save_audio("test_streaming.mp3", final_audio_np)
    print("\nStandard streaming test finished. Audio saved to test_streaming.mp3\n")
else:
    print("\nStandard streaming test produced no audio.\n")
