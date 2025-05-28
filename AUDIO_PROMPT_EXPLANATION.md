# How audio_prompt Works in Dia Model

## Overview

The `audio_prompt` parameter in the Dia model allows for **voice cloning** - using an audio sample to influence the voice characteristics of the generated speech. This enables the model to mimic the voice, tone, and speaking style from a reference audio file.

## How It Works

### 1. Audio Prompt Processing

When you provide an `audio_prompt` to the `generate()` method:

1. **Audio Loading**: The audio file is loaded and encoded using the DAC (Descript Audio Codec) model
   - Converted to 44.1kHz mono if needed
   - Encoded into discrete audio tokens (codebook indices)

2. **Prefill Phase**: The encoded audio tokens are used to "prefill" the decoder
   - These tokens provide context about voice characteristics
   - The model learns the speaking style, tone, and voice qualities from these tokens

3. **Generation**: The model continues generating new audio tokens that match:
   - The text content you want to synthesize
   - The voice characteristics from the audio prompt

### 2. Technical Implementation

From `dia/model.py`:

```python
def _prepare_audio_prompt(self, audio_prompts: list[torch.Tensor | None]) -> tuple[torch.Tensor, list[int]]:
    """Prepares the audio prompt tensor for the decoder.
    
    - Adds beginning-of-sequence (BOS) token
    - Applies delay pattern for multi-channel generation
    - Returns prefilled audio tokens and prefill steps
    """
```

The audio prompt influences generation through:
- **Decoder Prefill**: Audio prompt tokens are fed to the decoder before text generation begins
- **Cross-Attention**: The decoder attends to both text encoding and prefilled audio context
- **Conditional Generation**: The model generates audio that matches both text and voice style

### 3. Usage Examples

#### Basic Voice Cloning
```python
# Load a reference audio file
audio_prompt = model.load_audio("reference_voice.mp3")

# Generate speech with the cloned voice
output = model.generate(
    "[S1] Hello, this is a test.",
    audio_prompt=audio_prompt
)
```

#### Voice Cloning with Context (Recommended)
```python
# Provide the transcript of the reference audio
clone_text = "[S1] This is what the reference speaker said."
clone_audio = "reference_voice.mp3"

# Generate new content with the same voice
new_text = "[S1] This is new content in the same voice."
output = model.generate(
    clone_text + new_text,
    audio_prompt=clone_audio
)
```

### 4. FastAPI Server Implementation

The FastAPI server provides endpoints for voice cloning:

#### Upload Audio Prompt
```bash
curl -X POST "http://localhost:7860/audio_prompts/upload" \
  -F "prompt_id=my_voice" \
  -F "audio_file=@voice_sample.wav"
```

#### Create Voice Mapping
```bash
curl -X POST "http://localhost:7860/voice_mappings" \
  -H "Content-Type: application/json" \
  -d '{
    "voice_id": "custom_voice",
    "audio_prompt": "my_voice"
  }'
```

### 5. Important Considerations

1. **Audio Quality**: Better quality reference audio produces better voice cloning
   - Recommended: 3-10 seconds of clean speech
   - Avoid background noise or music

2. **Transcript Accuracy**: Including the transcript of the reference audio improves results
   - The model can better understand the voice-to-text mapping
   - Helps maintain consistency between voice and content

3. **Speaker Consistency**: The audio prompt primarily affects the speaker tagged in the text
   - If using `[S1]`, the audio prompt influences Speaker 1's voice
   - Multiple speakers can have different audio prompts

4. **Performance**: Using audio prompts increases generation time slightly
   - Additional encoding step for the reference audio
   - Larger context window during generation

## How Voice Characteristics Are Captured

The model captures several aspects from the audio prompt:

1. **Voice Timbre**: The unique quality of the voice
2. **Speaking Style**: Pace, rhythm, and articulation patterns
3. **Emotional Tone**: The emotional coloring of the speech
4. **Prosody**: Intonation and stress patterns
5. **Speaker Identity**: Gender, age, and individual voice characteristics

## Limitations

- Cannot perfectly clone all voices (especially very unique voices)
- Works best with voices similar to those in training data
- Requires clean, high-quality reference audio
- The generated voice may not be 100% identical but will have similar characteristics

## Best Practices

1. **Use High-Quality Audio**: Clean recordings without background noise
2. **Match Content Style**: Reference audio should have similar speaking style to target
3. **Appropriate Length**: 3-10 seconds is optimal for most use cases
4. **Include Transcript**: Always provide the transcript of reference audio when possible
5. **Test Different Samples**: Try multiple reference samples to find the best match