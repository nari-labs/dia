from transformers import AutoProcessor, DiaForConditionalGeneration


torch_device = "cuda"
# Example for Turkish TTS
# model_checkpoint = "nari-labs/Dia-1.6B-0626-tr"  # Assumed Turkish model ID
# text = [
#     "[S1] Merhaba, bu Türkçe bir test metnidir. [S2] Dia modeli ile Türkçe konuşma sentezleyebilirsiniz."
# ]

# Default English example
model_checkpoint = "nari-labs/Dia-1.6B-0626"
text = [
    "[S1] Dia is an open weights text to dialogue model. [S2] You get full control over scripts and voices. [S1] Wow. Amazing. (laughs) [S2] Try it now on Git hub or Hugging Face."
]


processor = AutoProcessor.from_pretrained(model_checkpoint)
inputs = processor(text=text, padding=True, return_tensors="pt").to(torch_device)

model = DiaForConditionalGeneration.from_pretrained(model_checkpoint).to(torch_device)
outputs = model.generate(**inputs, max_new_tokens=3072, guidance_scale=3.0, temperature=1.8, top_p=0.90, top_k=45)

outputs = processor.batch_decode(outputs)
processor.save_audio(outputs, "example_hf.mp3") # Renamed to avoid conflict with cli.py example
