import argparse
import os
import random

import numpy as np
import soundfile as sf
import torch

from dia.model import Dia


def set_seed(seed: int):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for cuDNN (if used)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description="Generate audio using the Dia model.")

    parser.add_argument("text", type=str, help="Input text for speech generation.")
    parser.add_argument(
        "--output", type=str, required=True, help="Path to save the generated audio file (e.g., output.wav)."
    )

    parser.add_argument(
        "--repo-id",
        type=str,
        default="nari-labs/Dia-1.6B",
        help="Hugging Face repository ID (e.g., nari-labs/Dia-1.6B).",
    )
    parser.add_argument(
        "--local-paths", action="store_true", help="Load model from local config and checkpoint files."
    )

    parser.add_argument(
        "--config", type=str, help="Path to local config.json file (required if --local-paths is set)."
    )
    parser.add_argument(
        "--checkpoint", type=str, help="Path to local model checkpoint .pth file (required if --local-paths is set)."
    )
    parser.add_argument(
        "--audio-prompt", type=str, default=None, help="Path to an optional audio prompt WAV file for voice cloning."
    )

    gen_group = parser.add_argument_group("Generation Parameters")
    gen_group.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum number of audio tokens to generate (defaults to config value).",
    )
    gen_group.add_argument(
        "--cfg-scale", type=float, default=3.0, help="Classifier-Free Guidance scale (default: 3.0)."
    )
    gen_group.add_argument(
        "--temperature", type=float, default=1.3, help="Sampling temperature (higher is more random, default: 0.7)."
    )
    gen_group.add_argument("--top-p", type=float, default=0.95, help="Nucleus sampling probability (default: 0.95).")

    infra_group = parser.add_argument_group("Infrastructure")
    infra_group.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    infra_group.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (e.g., 'cuda', 'cpu', default: auto).",
    )

    args = parser.parse_args()

    # Validation for local paths
    if args.local_paths:
        if not args.config:
            parser.error("--config is required when --local-paths is set.")
        if not args.checkpoint:
            parser.error("--checkpoint is required when --local-paths is set.")
        if not os.path.exists(args.config):
            parser.error(f"Config file not found: {args.config}")
        if not os.path.exists(args.checkpoint):
            parser.error(f"Checkpoint file not found: {args.checkpoint}")

    # Set seed if provided
    if args.seed is not None:
        set_seed(args.seed)
        print(f"Using user-selected seed: {args.seed}")

    # Determine device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load model
    print("Loading model...")
    if args.local_paths:
        print(f"Loading from local paths: config='{args.config}', checkpoint='{args.checkpoint}'")
        try:
            model = Dia.from_local(args.config, args.checkpoint, device=device)
        except Exception as e:
            print(f"Error loading local model: {e}")
            exit(1)
    else:
        print(f"Loading from Hugging Face Hub: repo_id='{args.repo_id}'")
        try:
            model = Dia.from_pretrained(args.repo_id, device=device)
        except Exception as e:
            print(f"Error loading model from Hub: {e}")
            exit(1)
    print("Model loaded.")

    # Generate audio
    print("Generating audio...")
    try:
        sample_rate = 44100  # Default assumption

        output_audio = model.generate(
            text=args.text,
            audio_prompt=args.audio_prompt,
            max_tokens=args.max_tokens,
            cfg_scale=args.cfg_scale,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print("Audio generation complete.")

        print(f"Saving audio to {args.output}...")
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

        sf.write(args.output, output_audio, sample_rate)
        print(f"Audio successfully saved to {args.output}")

    except Exception as e:
        print(f"Error during audio generation or saving: {e}")
        exit(1)


if __name__ == "__main__":
    main()
