# Dia on Replicate

<p align="center">
  <a href="https://replicate.com/zsxkib/dia"><img src="https://replicate.com/zsxkib/dia/badge" alt="Run on Replicate" height=38></a>
</p>

This directory contains the files to run the [Nari Labs Dia model](https://github.com/nari-labs/dia) on [Replicate](https://replicate.com).

- `cog.yaml`: Defines the Python environment, system packages, and GPU requirements for Replicate.
- `predict.py`: A script that implements the Cog `Predictor` interface to load the model and run predictions.

## Demo & API

You can try the demo and access the API here: [https://replicate.com/zsxkib/dia](https://replicate.com/zsxkib/dia)

I also posted about it on X/Twitter here: [https://x.com/zsakib_/status/1915037657064759716](https://x.com/zsakib_/status/1915037657064759716)

## Example API Usage

```python
import replicate

output = replicate.run(
    "zsxkib/dia:VERSION_HASH", # Replace VERSION_HASH with the actual model version hash
    input={
        "text": "[S1] Hello there! How are you? [S2] I'm doing great, thanks for asking! (laughs)",
        "audio_prompt": "https://replicate.delivery/pbxt/YourAudioFile.mp3", # Optional
        "max_audio_prompt_seconds": 10, # Optional, controls prompt length (default 10s)
        "temperature": 1.3,
        "cfg_scale": 3.0
        # ... other parameters ...
    }
)
print(output)
# Output: URL to the generated .wav file
```

On Replicate, users pay for their own compute time, which helps keep things running even if a model gets popular. 