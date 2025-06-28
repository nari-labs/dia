import argparse
import contextlib
import gc
import io
import random
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
import numpy as np
import soundfile as sf
import torch

from dia.model import Dia


# --- Global Setup ---
parser = argparse.ArgumentParser(description="Gradio interface for Nari TTS")
parser.add_argument("--device", type=str, default=None, help="Force device (e.g., 'cuda', 'mps', 'cpu')")
parser.add_argument("--share", action="store_true", help="Enable Gradio sharing")

args = parser.parse_args()


# Determine device
if args.device:
    device = torch.device(args.device)
elif torch.cuda.is_available():
    device = torch.device("cuda")
# Simplified MPS check for broader compatibility
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    # Basic check is usually sufficient, detailed check can be problematic
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# Load Nari model and config
print("Loading Nari model...")
try:
    dtype_map = {
        "cpu": "float32",
        "mps": "float32",  # Apple M series – better with float32
        "cuda": "float16",  # NVIDIA – better with float16
    }

    dtype = dtype_map.get(device.type, "float16")
    print(f"Using device: {device}, attempting to load model with {dtype}")
    # Step 1: Load model normally
    model = Dia.from_pretrained("nari-labs/Dia-1.6B", compute_dtype=dtype, device=device)

    # Step 2: Apply dynamic quantization
    quantized_model = torch.quantization.quantize_dynamic(
        model.model, {torch.nn.Linear, torch.nn.LSTM}, dtype=torch.qint8
    )

    # Step 3: Dereference the original
    model.model = None
    torch.cuda.empty_cache()

    # Step 4: Replace with quantized
    model.model = quantized_model

except Exception as e:
    print(f"Error loading Nari model: {e}")
    raise


def set_seed(seed: int):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_effective_length(text):
    """Counts effective length treating [S1] and [S2] as single characters."""
    return len(text.replace("[S1]", "¤").replace("[S2]", "¤"))


def auto_adjust_chunk_size(text, user_chunk_size):
    """Auto-adjusts chunk size if turbo mode is enabled."""
    effective_chars = count_effective_length(text)
    if user_chunk_size > 0:
        # If user explicitly sets a chunk size, respect it
        return int(user_chunk_size)
    else:
        # Auto-tune based on input size
        if effective_chars <= 1024:
            return 48
        elif effective_chars <= 4096:
            return 64
        else:
            return 96


def split_by_words_respecting_special_tokens(text, max_effective_chars=64):
    """Splits text into chunks close to max_effective_chars, preserving full words and [S1]/[S2] markers."""
    words = text.split()
    chunks = []
    current_chunk = ""

    for word in words:
        tentative_chunk = (current_chunk + " " + word).strip() if current_chunk else word
        if count_effective_length(tentative_chunk) > max_effective_chars:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = word
            else:
                chunks.append(word)
                current_chunk = ""
        else:
            current_chunk = tentative_chunk

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def batch_chunks(chunks, batch_size):
    """Yield successive batches of chunks."""
    for i in range(0, len(chunks), batch_size):
        yield chunks[i : i + batch_size]


def split_lines_greedy(lines, chunk_size):
    """Greedily split lines into chunks of up to chunk_size lines."""
    chunks = []
    i = 0
    while i < len(lines):
        remaining = len(lines) - i
        if remaining <= chunk_size:
            chunks.append("\n".join(lines[i:]))
            break
        else:
            chunks.append("\n".join(lines[i : i + chunk_size]))
            i += chunk_size
    return chunks


def set_seed(seed: int):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_inference(
    text_input: str,
    audio_prompt_text_input: str,
    audio_prompt_input: Optional[Tuple[int, np.ndarray]],
    max_new_tokens: int,
    cfg_scale: float,
    temperature: float,
    top_p: float,
    cfg_filter_top_k: int,
    speed_factor: float,
    chunk_size: int,
    seed: Optional[int] = None,
):
    """
    Runs Nari inference using the globally loaded model and provided inputs.
    Supports dynamic chunking and token scaling.
    """

    global model, device  # Access global model, config, device
    console_output_buffer = io.StringIO()

    with contextlib.redirect_stdout(console_output_buffer):
        # Validation
        if not text_input or text_input.isspace():
            raise gr.Error("Text input cannot be empty.")

        if audio_prompt_input and (not audio_prompt_text_input or audio_prompt_text_input.isspace()):
            raise gr.Error("Audio Prompt Text input cannot be empty.")

        # Set and Display Generation Seed
        if seed is None or seed < 0:
            seed = random.randint(0, 2**32 - 1)
            print(f"\nNo seed provided, generated random seed: {seed}\n")
        else:
            print(f"\nUsing user-selected seed: {seed}\n")
        set_seed(seed)

        # Preprocess audio prompt
        temp_audio_prompt_path = None
        output_audio = (44100, np.zeros(1, dtype=np.float32))
        prompt_path_for_generate = None

        try:
            if audio_prompt_input is not None:
                sr, audio_data = audio_prompt_input
                if audio_data is None or audio_data.size == 0 or audio_data.max() == 0:
                    gr.Warning("Audio prompt seems empty or silent, ignoring prompt.")
                else:
                    with tempfile.NamedTemporaryFile(mode="wb", suffix=".wav", delete=False) as f_audio:
                        temp_audio_prompt_path = f_audio.name

                        if np.issubdtype(audio_data.dtype, np.integer):
                            max_val = np.iinfo(audio_data.dtype).max
                            audio_data = audio_data.astype(np.float32) / max_val
                        elif not np.issubdtype(audio_data.dtype, np.floating):
                            try:
                                audio_data = audio_data.astype(np.float32)
                            except Exception as conv_e:
                                raise gr.Error(f"Failed to convert audio prompt: {conv_e}")

                        if audio_data.ndim > 1:
                            audio_data = np.mean(audio_data, axis=-1)
                            audio_data = np.ascontiguousarray(audio_data)

                        sf.write(temp_audio_prompt_path, audio_data, sr, subtype="FLOAT")
                        prompt_path_for_generate = temp_audio_prompt_path
                        print(f"Created temporary audio prompt file: {temp_audio_prompt_path} (orig sr: {sr})")

            # --- Chunking ---
            chunk_size = auto_adjust_chunk_size(text_input, chunk_size)
            print(f"Auto-selected chunk size: {chunk_size} effective characters per chunk.")
            # New: Split by effective character count (~64 chars per chunk)
            chunks = split_by_words_respecting_special_tokens(text_input, max_effective_chars=chunk_size)

            print(f"Chunked into {len(chunks)} chunks (based on effective character count).")

            audio_segments = []

            start_time = time.time()

            batch_size = 4  # Adjust based on your GPU VRAM (e.g., 2–8)

            for batch_idx, chunk_batch in enumerate(batch_chunks(chunks, batch_size)):
                print(
                    f"Generating batch {batch_idx + 1}/{(len(chunks) + batch_size - 1) // batch_size} with {len(chunk_batch)} chunks..."
                )

                batch_input_text = "\n".join(chunk.strip() for chunk in chunk_batch).strip()

                if not batch_input_text:
                    raise gr.Error("All chunks in this batch were empty after trimming. Cannot generate.")
                effective_chars = count_effective_length(batch_input_text)
                scaling_factor = effective_chars / chunk_size
                adjusted_tokens = int(max_new_tokens * scaling_factor)
                adjusted_tokens = max(256, adjusted_tokens)

                with torch.inference_mode(), torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    generated_batch_audio = model.generate(
                        batch_input_text,
                        max_tokens=adjusted_tokens,
                        cfg_scale=cfg_scale,
                        temperature=temperature,
                        top_p=top_p,
                        cfg_filter_top_k=cfg_filter_top_k,
                        use_torch_compile=False,
                        audio_prompt=prompt_path_for_generate,
                        audio_prompt_text=audio_prompt_text_input,
                    )

                if generated_batch_audio is not None:
                    audio_segments.append(generated_batch_audio)

                    # Add a small silence buffer **after the batch** (but NOT after the last batch)
                    if batch_idx < (len(chunks) + batch_size - 1) // batch_size - 1:
                        silence_duration_sec = 0.2
                        silence_samples = int(44100 * silence_duration_sec)
                        silence = np.zeros(silence_samples, dtype=np.float32)
                        audio_segments.append(silence)

            if not audio_segments:
                output_audio_np = None
            else:
                output_audio_np = np.concatenate(audio_segments)

            end_time = time.time()
            print(f"Generation finished in {end_time - start_time:.2f} seconds.\n")

            # --- Postprocessing ---
            if output_audio_np is not None:
                output_sr = 44100

                # Slowdown if needed
                original_len = len(output_audio_np)
                speed_factor = max(0.1, min(speed_factor, 5.0))
                target_len = int(original_len / speed_factor)

                if target_len != original_len and target_len > 0:
                    x_original = np.arange(original_len)
                    x_resampled = np.linspace(0, original_len - 1, target_len)
                    resampled_audio_np = np.interp(x_resampled, x_original, output_audio_np)
                    output_audio = (output_sr, resampled_audio_np.astype(np.float32))
                    print(
                        f"Resampled audio from {original_len} to {target_len} samples for {speed_factor:.2f}x speed."
                    )
                else:
                    output_audio = (output_sr, output_audio_np)
                    print(f"Skipping audio speed adjustment (factor: {speed_factor:.2f}).")

                # Final output conversion
                if output_audio[1].dtype in (np.float32, np.float64):
                    audio_for_gradio = np.clip(output_audio[1], -1.0, 1.0)
                    audio_for_gradio = (audio_for_gradio * 32767).astype(np.int16)
                    output_audio = (output_sr, audio_for_gradio)
                    print("Converted audio to int16 for Gradio output.")
            else:
                print("\nGeneration finished, but no valid tokens were produced.")
                gr.Warning("Generation produced no output.")

        except Exception as e:
            print(f"Error during inference: {e}")
            import traceback

            traceback.print_exc()
            raise gr.Error(f"Inference failed: {e}")

        finally:
            # Clean up temp files
            if temp_audio_prompt_path and Path(temp_audio_prompt_path).exists():
                try:
                    Path(temp_audio_prompt_path).unlink()
                    print(f"Deleted temporary audio prompt file: {temp_audio_prompt_path}")
                except Exception as cleanup_e:
                    print(f"Warning: Error deleting temporary audio prompt file: {cleanup_e}")

        console_output = console_output_buffer.getvalue()

    try:
        torch.cuda.empty_cache()
        gc.collect()
    except RuntimeError as e:
        print(f"CUDA cache clear failed: {e}")
    except Exception as e:
        print(f"Garbage collection failed: {e}")
    finally:
        print("Generation completed")

    return output_audio, seed, console_output


# --- Create Gradio Interface ---
css = """
#col-container {max-width: 90%; margin-left: auto; margin-right: auto;}
"""
# Attempt to load default text from example.txt
default_text = "[S1] Dia is an open weights text to dialogue model. \n[S2] You get full control over scripts and voices. \n[S1] Wow. Amazing. (laughs) \n[S2] Try it now on Git hub or Hugging Face."
example_txt_path = Path("./example.txt")
if example_txt_path.exists():
    try:
        default_text = example_txt_path.read_text(encoding="utf-8").strip()
        if not default_text:  # Handle empty example file
            default_text = "Example text file was empty."
    except Exception as e:
        print(f"Warning: Could not read example.txt: {e}")


# Build Gradio UI
with gr.Blocks(css=css, theme="gradio/dark") as demo:
    gr.Markdown("# Nari Text-to-Speech Synthesis")

    with gr.Row(equal_height=False):
        with gr.Column(scale=1):
            with gr.Accordion("Audio Reference Prompt (Optional)", open=False):
                audio_prompt_input = gr.Audio(
                    label="Audio Prompt (Optional)",
                    show_label=True,
                    sources=["upload", "microphone"],
                    type="numpy",
                )
                audio_prompt_text_input = gr.Textbox(
                    label="Transcript of Audio Prompt (Required if using Audio Prompt)",
                    placeholder="Enter text here...",
                    value="",
                    lines=5,  # Increased lines
                )
            text_input = gr.Textbox(
                label="Text To Generate",
                placeholder="Enter text here...",
                value=default_text,
                lines=5,  # Increased lines
            )
            with gr.Accordion("Generation Parameters", open=False):
                chunk_size = gr.Number(
                    label="Chunk Size (Effective Characters)",
                    minimum=0,
                    value=0,
                    precision=0,
                    step=1,
                    info="If 0, auto-selects chunk size for optimal speed. Otherwise, set number of effective characters per generation chunk.",
                )
                max_new_tokens = gr.Slider(
                    label="Max New Tokens (Audio Length)",
                    minimum=860,
                    maximum=3072,
                    value=model.config.decoder_config.max_position_embeddings,  # Use config default if available, else fallback
                    step=50,
                    info="Controls the maximum length of the generated audio (more tokens = longer audio).",
                )
                cfg_scale = gr.Slider(
                    label="CFG Scale (Guidance Strength)",
                    minimum=1.0,
                    maximum=5.0,
                    value=3.0,  # Default from inference.py
                    step=0.1,
                    info="Higher values increase adherence to the text prompt.",
                )
                temperature = gr.Slider(
                    label="Temperature (Randomness)",
                    minimum=1.0,
                    maximum=2.5,
                    value=1.8,  # Default from inference.py
                    step=0.05,
                    info="Lower values make the output more deterministic, higher values increase randomness.",
                )
                top_p = gr.Slider(
                    label="Top P (Nucleus Sampling)",
                    minimum=0.70,
                    maximum=1.0,
                    value=0.95,  # Default from inference.py
                    step=0.01,
                    info="Filters vocabulary to the most likely tokens cumulatively reaching probability P.",
                )
                cfg_filter_top_k = gr.Slider(
                    label="CFG Filter Top K",
                    minimum=15,
                    maximum=100,
                    value=45,
                    step=1,
                    info="Top k filter for CFG guidance.",
                )
                speed_factor_slider = gr.Slider(
                    label="Speed Factor",
                    minimum=0.8,
                    maximum=1.0,
                    value=1.0,
                    step=0.02,
                    info="Adjusts the speed of the generated audio (1.0 = original speed).",
                )
                seed_input = gr.Number(
                    label="Generation Seed (Optional)",
                    value=-1,
                    precision=0,  # No decimal points
                    step=1,
                    interactive=True,
                    info="Set a generation seed for reproducible outputs. Leave empty or -1 for random seed.",
                )

            run_button = gr.Button("Generate Audio", variant="primary")

        with gr.Column(scale=1):
            audio_output = gr.Audio(
                label="Generated Audio",
                type="numpy",
                autoplay=False,
            )
            seed_output = gr.Textbox(label="Generation Seed", interactive=False)
            console_output = gr.Textbox(label="Console Output Log", lines=10, interactive=False)

    # Link button click to function
    run_button.click(
        fn=run_inference,
        inputs=[
            text_input,
            audio_prompt_text_input,
            audio_prompt_input,
            max_new_tokens,
            cfg_scale,
            temperature,
            top_p,
            cfg_filter_top_k,
            speed_factor_slider,
            chunk_size,
            seed_input,
        ],
        outputs=[
            audio_output,
            seed_output,
            console_output,
        ],  # Add status_output here if using it
        api_name="generate_audio",
    )

    # Add examples (ensure the prompt path is correct or remove it if example file doesn't exist)
    example_prompt_path = "./example_prompt.mp3"  # Adjust if needed
    examples_list = [
        [
            "[S1] Oh fire! Oh my goodness! What's the procedure? What to we do people? The smoke could be coming through an air duct! \n[S2] Oh my god! Okay.. it's happening. Everybody stay calm! \n[S1] What's the procedure... \n[S2] Everybody stay fucking calm!!!... Everybody fucking calm down!!!!! \n[S1] No! No! If you touch the handle, if its hot there might be a fire down the hallway! ",
            None,
            3072,
            3.0,
            1.8,
            0.95,
            35,
            0.94,
            4,
            -1,
        ],
        [
            "[S1] Open weights text to dialogue model. \n[S2] You get full control over scripts and voices. \n[S1] I'm biased, but I think we clearly won. \n[S2] Hard to disagree. (laughs) \n[S1] Thanks for listening to this demo. \n[S2] Try it now on Git hub and Hugging Face. \n[S1] If you liked our model, please give us a star and share to your friends. \n[S2] This was Nari Labs.",
            example_prompt_path if Path(example_prompt_path).exists() else None,
            3072,
            3.0,
            1.8,
            0.95,
            35,
            0.94,
            4,
            -1,
        ],
    ]

    if examples_list:
        gr.Examples(
            examples=examples_list,
            inputs=[
                text_input,
                audio_prompt_input,
                max_new_tokens,
                cfg_scale,
                temperature,
                top_p,
                cfg_filter_top_k,
                speed_factor_slider,
                chunk_size,
                seed_input,
            ],
            outputs=[audio_output],
            fn=run_inference,
            cache_examples=False,
            label="Examples (Click to Run)",
        )
    else:
        gr.Markdown("_(No examples configured or example prompt file missing)_")

# --- Launch the App ---
if __name__ == "__main__":
    print("Launching Gradio interface...")

    # set `GRADIO_SERVER_NAME`, `GRADIO_SERVER_PORT` env vars to override default values
    # use `GRADIO_SERVER_NAME=0.0.0.0` for Docker
    demo.launch(share=args.share)
