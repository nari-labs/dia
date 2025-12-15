# Prediction interface for Cog ⚙️
# https://github.com/nari-labs/dia

import os
import subprocess

MODEL_CACHE = "model_cache"
BASE_URL = f"https://weights.replicate.delivery/default/dia/{MODEL_CACHE}/"
os.environ["HF_HOME"] = MODEL_CACHE
os.environ["TORCH_HOME"] = MODEL_CACHE
os.environ["HF_DATASETS_CACHE"] = MODEL_CACHE
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE

import time
import torch
import numpy as np
import soundfile as sf
import tempfile
from cog import BasePredictor, Input, Path
from typing import Optional
import random

# Import Dia model
from dia.model import Dia

OUTPUT_SAMPLE_RATE = 44100 # Dia model output sample rate

def download_weights(url: str, dest: str) -> None:
    start = time.time()
    print("[!] Initiating download from URL: ", url)
    print("[~] Destination path: ", dest)
    if ".tar" in dest:
        dest = os.path.dirname(dest)
    command = ["pget", "-vf" + ("x" if ".tar" in url else ""), url, dest]
    try:
        print(f"[~] Running command: {' '.join(command)}")
        subprocess.check_call(command, close_fds=False)
    except subprocess.CalledProcessError as e:
        print(
            f"[ERROR] Failed to download weights. Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}."
        )
        raise
    print("[+] Download completed in: ", time.time() - start, "seconds")

def set_seed(seed: int):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for cuDNN (if used)
    # Not strictly necessary for this model but good practice
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        print("Setting up predictor...")
        start_time = time.time()

        # Create model cache directory if it doesn't exist
        if not os.path.exists(MODEL_CACHE):
            os.makedirs(MODEL_CACHE)

        # Determine device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Download model weights
        model_files = [
            "models--nari-labs--Dia-1.6B.tar",
        ]

        for model_file in model_files:
            url = BASE_URL + model_file
            filename = url.split("/")[-1]
            dest_path = os.path.join(MODEL_CACHE, filename)
            if not os.path.exists(dest_path.replace(".tar", "")):
                download_weights(url, dest_path)

        # Load Dia model
        print("Loading Dia model...")
        self.model = Dia.from_pretrained("nari-labs/Dia-1.6B", device=self.device)
        
        end_time = time.time()
        print(f"Setup complete in {end_time - start_time:.2f} seconds.")

    def predict(
        self,
        text: str = Input(description="Input text for dialogue generation. Use [S1], [S2] to indicate different speakers and (description) in parentheses for non-verbal cues e.g., (laughs), (whispers)."),
        audio_prompt: Optional[Path] = Input(description="Optional audio file (.wav/.mp3/.flac) for voice cloning. The model will attempt to mimic this voice style.", default=None),
        max_new_tokens: int = Input(
            description="Controls the length of generated audio. Higher values create longer audio. (86 tokens ≈ 1 second of audio).",
            default=3072,
            ge=500,
            le=4096
        ),
        max_audio_prompt_seconds: int = Input(
            description="Maximum duration in seconds for the input voice cloning audio prompt. Only used when an audio prompt is provided. Longer voice samples will be truncated to this length.",
            default=10,
            ge=1,
            le=120
        ),
        cfg_scale: float = Input(
            description="Controls how closely the audio follows your text. Higher values (3-5) follow text more strictly; lower values may sound more natural but deviate more.",
            default=3.0,
            ge=1.0,
            le=5.0
        ),
        temperature: float = Input(
            description="Controls randomness in generation. Higher values (1.3-2.0) increase variety; lower values make output more consistent. Set to 0 for deterministic (greedy) generation.",
            default=1.3,
            ge=0.0,
            le=2.0
        ),
        top_p: float = Input(
            description="Controls diversity of word choice. Higher values include more unusual options. Most users shouldn't need to adjust this parameter.",
            default=0.95,
            ge=0.1,
            le=1.0
        ),
        cfg_filter_top_k: int = Input(
            description="Technical parameter for filtering audio generation tokens. Higher values allow more diverse sounds; lower values create more consistent audio.",
            default=35,
            ge=10,
            le=100
        ),
        speed_factor: float = Input(
            description="Adjusts playback speed of the generated audio. Values below 1.0 slow down the audio; 1.0 is original speed.",
            default=0.94,
            ge=0.5,
            le=1.5
        ),
        seed: Optional[int] = Input(
            description="Random seed for reproducible results. Use the same seed value to get the same output for identical inputs. Leave blank for random results each time.",
            default=None
        ),
    ) -> Path:
        """Generate dialogue audio from text, optionally cloning voice from an audio prompt."""
        # Set random seed
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        set_seed(seed)

        # Validate text input
        if not text or text.isspace():
            raise ValueError("Text input cannot be empty.")

        # Handle audio prompt if provided
        temp_audio_prompt_path = None
        if audio_prompt is not None:
            if not str(audio_prompt).lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".opus")):
                print(f"Warning: Audio prompt file extension doesn't look like audio: {audio_prompt}. Trying anyway.")
            
            # Load audio and process to mono float32
            audio_data, sr = sf.read(str(audio_prompt), dtype='float32')
            print(f"Loaded audio prompt with shape: {audio_data.shape}, sample rate: {sr}")

            # Truncate audio prompt to max_audio_prompt_seconds
            max_prompt_samples = int(max_audio_prompt_seconds * sr)
            if audio_data.shape[0] > max_prompt_samples:
                print(f"Audio prompt is longer than {max_audio_prompt_seconds}s, truncating to {max_prompt_samples} samples.")
                audio_data = audio_data[:max_prompt_samples]
                print(f"Truncated audio prompt shape: {audio_data.shape}")
            # --- End Truncation ---

            # Ensure mono
            if audio_data.ndim > 1:
                if audio_data.shape[1] == 2: # Shape (N, 2)
                    print("Audio prompt is stereo, converting to mono by averaging channels.")
                    audio_data = np.mean(audio_data, axis=1)
                elif audio_data.shape[0] == 2: # Shape (2, N) - less common but handle anyway
                        print("Audio prompt is stereo (2, N), converting to mono by averaging channels.")
                        audio_data = np.mean(audio_data, axis=0)
                else:
                        print(f"Warning: Audio prompt has unexpected shape {audio_data.shape}. Attempting to use the first channel.")
                        # Fallback: take the first channel if shape is unusual
                        audio_data = audio_data[:, 0] if audio_data.shape[1] < audio_data.shape[0] else audio_data[0]

            # Ensure contiguous array after potential slicing/averaging
            audio_data = np.ascontiguousarray(audio_data)

            # Save processed audio to temporary WAV file
            temp_audio_prompt_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
            sf.write(temp_audio_prompt_path, audio_data, sr, subtype='FLOAT')
            print(f"Processed audio prompt saved to temporary file: {temp_audio_prompt_path}")

        # Generate audio
        print("Generating audio tokens...")
        start_time = time.time()
        with torch.inference_mode():
            output_audio_np = self.model.generate(
                text=text,
                audio_prompt_path=temp_audio_prompt_path,
                max_tokens=max_new_tokens,
                cfg_scale=cfg_scale,
                temperature=temperature,
                top_p=top_p,
                use_cfg_filter=True,
                cfg_filter_top_k=cfg_filter_top_k,
                use_torch_compile=False,
            )
        gen_end_time = time.time()
        print(f"Token generation finished in {gen_end_time - start_time:.2f} seconds.")

        # Clean up temporary audio prompt file if used
        if temp_audio_prompt_path and os.path.exists(temp_audio_prompt_path):
            os.unlink(temp_audio_prompt_path)

        # Validate output
        if output_audio_np is None or output_audio_np.size == 0:
            raise RuntimeError("Model generation failed to produce audio.")

        print(f"Generated audio shape: {output_audio_np.shape}")

        # Adjust speed
        original_len = len(output_audio_np)
        speed_factor = max(0.1, min(speed_factor, 5.0))  # Clamp speed factor
        target_len = int(original_len / speed_factor)

        if target_len != original_len and target_len > 0:
            print(f"Adjusting speed by factor {speed_factor:.2f}...")
            x_original = np.arange(original_len)
            x_resampled = np.linspace(0, original_len - 1, target_len)
            resampled_audio_np = np.interp(x_resampled, x_original, output_audio_np)
            final_audio_np = resampled_audio_np.astype(np.float32)
            print(f"Resampled audio from {original_len} to {target_len} samples.")
        else:
            final_audio_np = output_audio_np  # Keep original if no change or invalid calc
            print(f"Skipping audio speed adjustment (factor: {speed_factor:.2f}).")

        # Save output
        output_path = Path(tempfile.mkdtemp()) / "output.wav"
        print(f"Saving audio to {output_path}...")
        sf.write(str(output_path), final_audio_np, OUTPUT_SAMPLE_RATE, subtype='FLOAT')

        save_end_time = time.time()
        print(f"Audio saved in {save_end_time - gen_end_time:.2f} seconds.")
        print(f"Total prediction time: {save_end_time - start_time:.2f} seconds.")

        return output_path
