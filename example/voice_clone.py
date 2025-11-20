import numpy as np

from dia.model import Dia


model = Dia.from_pretrained("nari-labs/Dia-1.6B-0626", compute_dtype="float16")

# You should put the transcript of the voice you want to clone
# We will use the audio created by running simple.py as an example.
# Note that you will be REQUIRED TO RUN simple.py for the script to work as-is.
clone_from_text = "[S1] Dia is an open weights text to dialogue model. [S2] You get full control over scripts and voices. [S1] Wow. Amazing. (laughs) [S2] Try it now on Git hub or Hugging Face."
clone_from_audio = "simple.mp3"

# For your custom needs, replace above with below and add your audio file to this directory:
# clone_from_text = "[S1] ... [S2] ... [S1] ... corresponding to your_audio_name.mp3"
# clone_from_audio = "your_audio_name.mp3"

# Text to generate
text_to_generate = "[S1] Hello, how are you? [S2] I'm good, thank you. [S1] What's your name? [S2] My name is Dia. [S1] Nice to meet you. [S2] Nice to meet you too."

print("--- Running Test Case 1: Standard Generation with voice clone ---")
# It will only return the audio from the text_to_generate
output = model.generate(
    clone_from_text + text_to_generate,
    audio_prompt=clone_from_audio,
    use_torch_compile=False,
    verbose=True,
    cfg_scale=4.0,
    temperature=1.8,
    top_p=0.90,
    cfg_filter_top_k=50,
)

model.save_audio("voice_clone.mp3", output)

print("Audio saved to voice_clone.mp3")

print("--- Running Test Case 2: Standard Streaming with voice clone ---")

stream_generator = model.generate_streaming(
    clone_from_text + text_to_generate,
    audio_prompt=clone_from_audio,
    use_torch_compile=False,
    chunk_size=1024,
    temperature=1.8,
    cfg_scale=4.0,
    top_p=0.90,
    cfg_filter_top_k=50,
)

full_audio = []
for chunk in stream_generator:
    print(f"Received audio chunk of shape: {chunk.shape}")
    full_audio.append(chunk)

if full_audio:
    final_audio_np = np.concatenate(full_audio, axis=0)
    model.save_audio("test_streaming.mp3", final_audio_np)
    print("\nStandard streaming test finished. Audio saved to test_streaming.mp3\n")
else:
    print("\nStandard streaming test produced no audio.\n")
