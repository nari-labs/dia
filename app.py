import argparse
import tempfile
import time
import queue
import threading
import os
import io
import base64
from pathlib import Path
from typing import Optional, Tuple, List, Generator, Dict, Any

import gradio as gr
import numpy as np
import soundfile as sf
import torch

from dia.model import Dia


# --- Global Setup ---
parser = argparse.ArgumentParser(description="Gradio interface for Nari TTS")
parser.add_argument(
    "--device", type=str, default=None, help="Force device (e.g., 'cuda', 'mps', 'cpu')"
)
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
    # Use the function from inference.py
    model = Dia.from_pretrained("nari-labs/Dia-1.6B", device=device)
except Exception as e:
    print(f"Error loading Nari model: {e}")
    raise


def create_audio_html(audio_data, sample_rate=44100):
    """
    Create an HTML audio element with the audio data
    """
    # Convert audio to 16-bit PCM WAV
    audio_int16 = (audio_data * 32767).astype(np.int16)
    
    # Create in-memory WAV file
    wav_bytes = io.BytesIO()
    with sf.SoundFile(wav_bytes, mode='w', samplerate=sample_rate, channels=1, format='WAV', subtype='PCM_16') as f:
        f.write(audio_int16)
    
    # Encode as base64
    wav_bytes.seek(0)
    base64_audio = base64.b64encode(wav_bytes.read()).decode('utf-8')
    
    # Create HTML audio element with autoplay
    audio_html = f"""
    <audio autoplay controls style="width: 100%;">
        <source src="data:audio/wav;base64,{base64_audio}" type="audio/wav">
        Your browser does not support the audio element.
    </audio>
    """
    return audio_html


def run_inference(
    text_input: str,
    audio_prompt_input: Optional[Tuple[int, np.ndarray]],
    max_new_tokens: int,
    cfg_scale: float,
    temperature: float,
    top_p: float,
    cfg_filter_top_k: int,
    speed_factor: float,
    use_streaming: bool = False,
    autoplay_chunks: bool = False,
    chunk_size: int = 100,
    overlap: int = 10,
    progress=gr.Progress(),
):
    """
    Runs Nari inference using the globally loaded model and provided inputs.
    Uses temporary files for text and audio prompt compatibility with inference.generate.
    """
    global model, device  # Access global model, config, device

    if not text_input or text_input.isspace():
        raise gr.Error("Text input cannot be empty.")
        
    # Show different progress message based on mode
    if use_streaming:
        progress(0, desc="Preparing streaming generation...")
    else:
        progress(0, desc="Preparing generation...")

    temp_txt_file_path = None
    temp_audio_prompt_path = None
    output_audio = (44100, np.zeros(1, dtype=np.float32))
    
    # For streaming with real-time playback
    audio_element_html = ""
    streaming_outputs = []

    try:
        prompt_path_for_generate = None
        if audio_prompt_input is not None:
            sr, audio_data = audio_prompt_input
            # Check if audio_data is valid
            if (
                audio_data is None or audio_data.size == 0 or audio_data.max() == 0
            ):  # Check for silence/empty
                gr.Warning("Audio prompt seems empty or silent, ignoring prompt.")
            else:
                # Save prompt audio to a temporary WAV file
                with tempfile.NamedTemporaryFile(
                    mode="wb", suffix=".wav", delete=False
                ) as f_audio:
                    temp_audio_prompt_path = f_audio.name  # Store path for cleanup

                    # Basic audio preprocessing for consistency
                    # Convert to float32 in [-1, 1] range if integer type
                    if np.issubdtype(audio_data.dtype, np.integer):
                        max_val = np.iinfo(audio_data.dtype).max
                        audio_data = audio_data.astype(np.float32) / max_val
                    elif not np.issubdtype(audio_data.dtype, np.floating):
                        gr.Warning(
                            f"Unsupported audio prompt dtype {audio_data.dtype}, attempting conversion."
                        )
                        # Attempt conversion, might fail for complex types
                        try:
                            audio_data = audio_data.astype(np.float32)
                        except Exception as conv_e:
                            raise gr.Error(
                                f"Failed to convert audio prompt to float32: {conv_e}"
                            )

                    # Ensure mono (average channels if stereo)
                    if audio_data.ndim > 1:
                        if audio_data.shape[0] == 2:  # Assume (2, N)
                            audio_data = np.mean(audio_data, axis=0)
                        elif audio_data.shape[1] == 2:  # Assume (N, 2)
                            audio_data = np.mean(audio_data, axis=1)
                        else:
                            gr.Warning(
                                f"Audio prompt has unexpected shape {audio_data.shape}, taking first channel/axis."
                            )
                            audio_data = (
                                audio_data[0]
                                if audio_data.shape[0] < audio_data.shape[1]
                                else audio_data[:, 0]
                            )
                        audio_data = np.ascontiguousarray(
                            audio_data
                        )  # Ensure contiguous after slicing/mean

                    # Write using soundfile
                    try:
                        sf.write(
                            temp_audio_prompt_path, audio_data, sr, subtype="FLOAT"
                        )  # Explicitly use FLOAT subtype
                        prompt_path_for_generate = temp_audio_prompt_path
                        print(
                            f"Created temporary audio prompt file: {temp_audio_prompt_path} (orig sr: {sr})"
                        )
                    except Exception as write_e:
                        print(f"Error writing temporary audio file: {write_e}")
                        raise gr.Error(f"Failed to save audio prompt: {write_e}")

        # 3. Run Generation
        start_time = time.time()
        
        # With torch.inference_mode() context manager for the generation call
        with torch.inference_mode():
            if use_streaming:
                progress(0.05, desc="Starting streaming generation...")
                
                # Collect audio chunks
                audio_chunks = []
                
                # Stream generation with progress updates
                total_chunks_estimate = (max_new_tokens // chunk_size) + 1
                chunk_count = 0
                
                # Status message for streaming
                if autoplay_chunks:
                    streaming_status = "Audio chunks will play as they're generated. The full audio will be available when generation completes."
                else:
                    streaming_status = "Generating audio in streaming mode. The full audio will be available when generation completes."
                
                # Always yield all three outputs - even if empty for now
                streaming_outputs = [None, streaming_status, ""]
                yield streaming_outputs
                
                # Use the streaming generator
                for i, audio_chunk in enumerate(model.generate_streaming(
                    text=text_input,
                    max_tokens=max_new_tokens,
                    cfg_scale=cfg_scale,
                    temperature=temperature,
                    top_p=top_p,
                    use_cfg_filter=True,
                    cfg_filter_top_k=cfg_filter_top_k,
                    use_torch_compile=False,  # Keep False for Gradio stability
                    audio_prompt_path=prompt_path_for_generate,
                    chunk_size=chunk_size,
                    overlap=overlap
                )):
                    # Store the chunk
                    audio_chunks.append(audio_chunk)
                    chunk_count += 1
                    
                    # Update progress
                    progress_value = min(0.05 + (0.9 * chunk_count / total_chunks_estimate), 0.95)
                    progress(progress_value, desc=f"Generated chunk {chunk_count}...")
                    
                    # Process the audio segment if autoplay is enabled
                    if autoplay_chunks:
                        # Apply speed factor to the chunk
                        current_len = len(audio_chunk)
                        target_len = int(current_len / speed_factor)
                        
                        if target_len != current_len and target_len > 0:
                            x_original = np.arange(current_len)
                            x_resampled = np.linspace(0, current_len - 1, target_len)
                            resampled_chunk = np.interp(x_resampled, x_original, audio_chunk)
                        else:
                            resampled_chunk = audio_chunk
                        
                        # Create auto-playing HTML audio element
                        audio_element_html = create_audio_html(resampled_chunk)
                        
                        # Update status message with chunk info
                        streaming_status = f"Playing chunk {chunk_count} of approximately {total_chunks_estimate} (estimated). Full audio will be available when generation completes."
                        
                        # Yield partial results to update the UI
                        streaming_outputs = [None, streaming_status, audio_element_html]
                        yield streaming_outputs
                    else:
                        # Still need to yield updates even when not auto-playing
                        streaming_status = f"Generated chunk {chunk_count} of approximately {total_chunks_estimate} (estimated). Full audio will be available when generation completes."
                        streaming_outputs = [None, streaming_status, ""]
                        yield streaming_outputs
                
                # Combine all chunks
                output_audio_np = np.concatenate(audio_chunks)
                progress(0.95, desc="Processing final audio...")
                
                # Log completion
                print(f"Streaming generation finished with {chunk_count} chunks")
                streaming_status = f"Generation complete with {chunk_count} chunks. Total generation time: {time.time() - start_time:.2f} seconds."
                # Always include the third parameter (empty HTML string)
                streaming_outputs = [None, streaming_status, ""]
                yield streaming_outputs
            else:
                progress(0.1, desc="Generating audio...")
                # Use regular generation
                output_audio_np = model.generate(
                    text=text_input,
                    max_tokens=max_new_tokens,
                    cfg_scale=cfg_scale,
                    temperature=temperature,
                    top_p=top_p,
                    use_cfg_filter=True,
                    cfg_filter_top_k=cfg_filter_top_k,
                    use_torch_compile=False,  # Keep False for Gradio stability
                    audio_prompt_path=prompt_path_for_generate,
                )
                progress(0.9, desc="Post-processing audio...")

        end_time = time.time()
        print(f"Generation finished in {end_time - start_time:.2f} seconds.")

        # 4. Convert Codes to Audio
        if output_audio_np is not None:
            # Get sample rate from the loaded DAC model
            output_sr = 44100

            # --- Slow down audio ---
            original_len = len(output_audio_np)
            # Ensure speed_factor is positive and not excessively small/large to avoid issues
            speed_factor = max(0.1, min(speed_factor, 5.0))
            target_len = int(
                original_len / speed_factor
            )  # Target length based on speed_factor
            if (
                target_len != original_len and target_len > 0
            ):  # Only interpolate if length changes and is valid
                x_original = np.arange(original_len)
                x_resampled = np.linspace(0, original_len - 1, target_len)
                resampled_audio_np = np.interp(x_resampled, x_original, output_audio_np)
                output_audio = (
                    output_sr,
                    resampled_audio_np.astype(np.float32),
                )  # Use resampled audio
                print(
                    f"Resampled audio from {original_len} to {target_len} samples for {speed_factor:.2f}x speed."
                )
            else:
                output_audio = (
                    output_sr,
                    output_audio_np,
                )  # Keep original if calculation fails or no change
                print(f"Skipping audio speed adjustment (factor: {speed_factor:.2f}).")
            # --- End slowdown ---

            print(
                f"Audio conversion successful. Final shape: {output_audio[1].shape}, Sample Rate: {output_sr}"
            )
            
            progress(1.0, desc="Done!")
            
            # Clear streaming status for final result
            if use_streaming:
                streaming_outputs = [output_audio, f"Generation complete in {end_time - start_time:.2f} seconds", ""]
                yield streaming_outputs
            else:
                return output_audio

        else:
            print("\nGeneration finished, but no valid tokens were produced.")
            # Return default silence
            gr.Warning("Generation produced no output.")
            if use_streaming:
                streaming_outputs = [output_audio, "Generation produced no output", ""]
                yield streaming_outputs
            else:
                return output_audio

    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback

        traceback.print_exc()
        # Re-raise as Gradio error to display nicely in the UI
        error_message = f"Inference failed: {e}"
        if use_streaming:
            streaming_outputs = [None, error_message, ""]
            yield streaming_outputs
        raise gr.Error(error_message)

    finally:
        # 5. Cleanup Temporary Files defensively
        if temp_txt_file_path and Path(temp_txt_file_path).exists():
            try:
                Path(temp_txt_file_path).unlink()
                print(f"Deleted temporary text file: {temp_txt_file_path}")
            except OSError as e:
                print(
                    f"Warning: Error deleting temporary text file {temp_txt_file_path}: {e}"
                )
        if temp_audio_prompt_path and Path(temp_audio_prompt_path).exists():
            try:
                Path(temp_audio_prompt_path).unlink()
                print(f"Deleted temporary audio prompt file: {temp_audio_prompt_path}")
            except OSError as e:
                print(
                    f"Warning: Error deleting temporary audio prompt file {temp_audio_prompt_path}: {e}"
                )

    if not use_streaming:
        return output_audio


# --- Create Gradio Interface ---
css = """
#col-container {max-width: 90%; margin-left: auto; margin-right: auto;}
.streaming-audio-player {
    margin-top: 10px;
    padding: 10px;
    border-radius: 8px;
    background-color: #f0f9ff;
    border: 1px solid #93c5fd;
}
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
with gr.Blocks(css=css) as demo:
    gr.Markdown("# Nari Text-to-Speech Synthesis")

    with gr.Row(equal_height=False):
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label="Input Text",
                placeholder="Enter text here...",
                value=default_text,
                lines=5,  # Increased lines
            )
            audio_prompt_input = gr.Audio(
                label="Audio Prompt (Optional)",
                show_label=True,
                sources=["upload", "microphone"],
                type="numpy",
            )
            
            with gr.Row():
                with gr.Column(scale=1):
                    streaming_toggle = gr.Checkbox(
                        label="Enable Streaming Generation",
                        value=False,
                        info="Generate audio in chunks for faster feedback"
                    )
                with gr.Column(scale=1):
                    autoplay_toggle = gr.Checkbox(
                        label="Auto-play Chunks",
                        value=True,
                        info="Play audio chunks as they're generated",
                        visible=False
                    )
            
            run_button = gr.Button("Generate Audio", variant="primary")
            
            with gr.Accordion("Generation Parameters", open=False):
                max_new_tokens = gr.Slider(
                    label="Max New Tokens (Audio Length)",
                    minimum=860,
                    maximum=3072,
                    value=model.config.data.audio_length,  # Use config default if available, else fallback
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
                    maximum=1.5,
                    value=1.3,  # Default from inference.py
                    step=0.05,
                    info="Lower values make the output more deterministic, higher values increase randomness.",
                )
                top_p = gr.Slider(
                    label="Top P (Nucleus Sampling)",
                    minimum=0.80,
                    maximum=1.0,
                    value=0.95,  # Default from inference.py
                    step=0.01,
                    info="Filters vocabulary to the most likely tokens cumulatively reaching probability P.",
                )
                cfg_filter_top_k = gr.Slider(
                    label="CFG Filter Top K",
                    minimum=15,
                    maximum=50,
                    value=30,
                    step=1,
                    info="Top k filter for CFG guidance.",
                )
                speed_factor_slider = gr.Slider(
                    label="Speed Factor",
                    minimum=0.8,
                    maximum=1.0,
                    value=0.94,
                    step=0.02,
                    info="Adjusts the speed of the generated audio (1.0 = original speed).",
                )
            
            with gr.Accordion("Streaming Parameters", open=False, visible=False) as streaming_accordion:
                chunk_size_slider = gr.Slider(
                    label="Chunk Size",
                    minimum=50,
                    maximum=300,
                    value=100,
                    step=10,
                    info="Number of tokens to generate per chunk (smaller = faster first output, but may have more artifacts).",
                )
                overlap_slider = gr.Slider(
                    label="Chunk Overlap",
                    minimum=0,
                    maximum=30,
                    value=10,
                    step=5,
                    info="Overlap between consecutive chunks (higher = smoother transitions, but slower).",
                )

        with gr.Column(scale=1):
            audio_output = gr.Audio(
                label="Generated Audio",
                type="numpy",
                autoplay=False,
                elem_id="main-audio-output"
            )
            status_output = gr.Markdown("Ready to generate audio")
            streaming_player = gr.HTML(
                visible=False,
                elem_classes="streaming-audio-player",
                elem_id="streaming-player"
            )

    # Make streaming parameters and autoplay toggle visible only when streaming is enabled
    streaming_toggle.change(
        fn=lambda x: [gr.update(visible=x), gr.update(visible=x)],
        inputs=[streaming_toggle],
        outputs=[streaming_accordion, autoplay_toggle]
    )

    # Link button click to function
    run_button.click(
        fn=run_inference,
        inputs=[
            text_input,
            audio_prompt_input,
            max_new_tokens,
            cfg_scale,
            temperature,
            top_p,
            cfg_filter_top_k,
            speed_factor_slider,
            streaming_toggle,
            autoplay_toggle,
            chunk_size_slider,
            overlap_slider,
        ],
        outputs=[audio_output, status_output, streaming_player],
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
            1.3,
            0.95,
            35,
            0.94,
            False,  # Streaming off
            True,   # Autoplay on
            100,
            10,
        ],
        [
            "[S1] Open weights text to dialogue model. \n[S2] You get full control over scripts and voices. \n[S1] I'm biased, but I think we clearly won. \n[S2] Hard to disagree. (laughs) \n[S1] Thanks for listening to this demo. \n[S2] Try it now on Git hub and Hugging Face. \n[S1] If you liked our model, please give us a star and share to your friends. \n[S2] This was Nari Labs.",
            example_prompt_path if Path(example_prompt_path).exists() else None,
            3072,
            3.0,
            1.3,
            0.95,
            35,
            0.94,
            True,  # Streaming on
            True,  # Autoplay on
            100,
            10,
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
                streaming_toggle,
                autoplay_toggle,
                chunk_size_slider,
                overlap_slider,
            ],
            outputs=[audio_output, status_output, streaming_player],
            fn=run_inference,
            cache_examples=False,
            label="Examples (Click to Run)",
        )
    else:
        gr.Markdown("_(No examples configured or example prompt file missing)_")


# --- Launch the App ---
if __name__ == "__main__":
    print("Launching Gradio interface...")
    demo.launch(share=args.share)
