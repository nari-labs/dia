import time
from enum import Enum
from typing import Callable, Generator

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from .audio import (
    apply_audio_delay,
    build_delay_indices,
    build_revert_indices,
    revert_audio_delay,
)
from .config import DiaConfig
from .layers import DiaModel
from .state import DecoderInferenceState, DecoderOutput, EncoderInferenceState


DEFAULT_SAMPLE_RATE = 44100
SAMPLE_RATE_RATIO = 512


def _get_default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _sample_next_token(
    logits_BCxV: torch.Tensor,
    temperature: float,
    top_p: float,
    top_k: int | None,
    audio_eos_value: int,
) -> torch.Tensor:
    if temperature == 0.0:
        return torch.argmax(logits_BCxV, dim=-1)

    logits_BCxV = logits_BCxV / temperature

    if audio_eos_value is not None and audio_eos_value >= 0:
        top_logit_indices_BC = torch.argmax(logits_BCxV, dim=-1)
        eos_not_highest_mask_BC = top_logit_indices_BC != audio_eos_value
        mask_eos_unless_highest_BCxV = torch.zeros_like(logits_BCxV, dtype=torch.bool)
        mask_eos_unless_highest_BCxV[eos_not_highest_mask_BC, audio_eos_value] = True
        logits_BCxV = logits_BCxV.masked_fill(mask_eos_unless_highest_BCxV, -torch.inf)
        eos_highest_mask_BC = top_logit_indices_BC == audio_eos_value
        mask_eos_highest_BCxV = torch.zeros_like(logits_BCxV, dtype=torch.bool)
        mask_eos_highest_BCxV[eos_highest_mask_BC, :audio_eos_value] = True
        logits_BCxV = logits_BCxV.masked_fill(mask_eos_highest_BCxV, -torch.inf)

    if top_k is not None:
        _, top_k_indices_BCxV = torch.topk(logits_BCxV, k=top_k, dim=-1)
        mask = torch.ones_like(logits_BCxV, dtype=torch.bool)
        mask = mask.scatter(dim=-1, index=top_k_indices_BCxV, value=False)
        logits_BCxV = logits_BCxV.masked_fill(mask, -torch.inf)

    if top_p < 1.0:
        probs_BCxV = torch.softmax(logits_BCxV, dim=-1)
        sorted_probs_BCxV, sorted_indices_BCxV = torch.sort(probs_BCxV, dim=-1, descending=True)
        cumulative_probs_BCxV = torch.cumsum(sorted_probs_BCxV, dim=-1)

        sorted_indices_to_remove_BCxV = cumulative_probs_BCxV > top_p
        sorted_indices_to_remove_BCxV = torch.roll(sorted_indices_to_remove_BCxV, shifts=1, dims=-1)
        sorted_indices_to_remove_BCxV[..., 0] = torch.zeros_like(sorted_indices_to_remove_BCxV[..., 0])

        indices_to_remove_BCxV = torch.zeros_like(sorted_indices_to_remove_BCxV)
        indices_to_remove_BCxV = indices_to_remove_BCxV.scatter(
            dim=-1, index=sorted_indices_BCxV, src=sorted_indices_to_remove_BCxV
        )
        logits_BCxV = logits_BCxV.masked_fill(indices_to_remove_BCxV, -torch.inf)

    final_probs_BCxV = torch.softmax(logits_BCxV, dim=-1)

    sampled_indices_BC = torch.multinomial(final_probs_BCxV, num_samples=1)
    sampled_indices_C = sampled_indices_BC.squeeze(-1)
    return sampled_indices_C


class ComputeDtype(str, Enum):
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"

    def to_dtype(self) -> torch.dtype:
        if self == ComputeDtype.FLOAT32:
            return torch.float32
        elif self == ComputeDtype.FLOAT16:
            return torch.float16
        elif self == ComputeDtype.BFLOAT16:
            return torch.bfloat16
        else:
            raise ValueError(f"Unsupported compute dtype: {self}")


class Dia:
    def __init__(
        self,
        config: DiaConfig,
        compute_dtype: str | ComputeDtype = ComputeDtype.FLOAT32,
        device: torch.device | None = None,
        load_dac: bool = True,
    ):
        """Initializes the Dia model.

        Args:
            config: The configuration object for the model.
            compute_dtype: The computation dtype to use.
            device: The device to load the model onto. If None, will automatically select the best available device.
            load_dac: Whether to load the DAC model.

        Raises:
            RuntimeError: If there is an error loading the DAC model.
        """
        super().__init__()
        self.config = config
        self.device = device if device is not None else _get_default_device()
        if isinstance(compute_dtype, str):
            compute_dtype = ComputeDtype(compute_dtype)
        self.compute_dtype = compute_dtype.to_dtype()
        self.model: DiaModel = DiaModel(config, self.compute_dtype)
        self.dac_model = None
        self._compiled_step = None
        self.load_dac = load_dac

        if not self.load_dac:
            print("Warning: DAC model will not be loaded. This is not recommended.")

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True

    @classmethod
    def from_local(
        cls,
        config_path: str,
        checkpoint_path: str,
        compute_dtype: str | ComputeDtype = ComputeDtype.FLOAT32,
        device: torch.device | None = None,
        load_dac: bool = True,
    ) -> "Dia":
        """Loads the Dia model from local configuration and checkpoint files.

        Args:
            config_path: Path to the configuration JSON file.
            checkpoint_path: Path to the model checkpoint (.pth) file.
            compute_dtype: The computation dtype to use.
            device: The device to load the model onto. If None, will automatically select the best available device.
            load_dac: Whether to load the DAC model.

        Returns:
            An instance of the Dia model loaded with weights and set to eval mode.

        Raises:
            FileNotFoundError: If the config or checkpoint file is not found.
            RuntimeError: If there is an error loading the checkpoint.
        """
        config = DiaConfig.load(config_path)
        if config is None:
            raise FileNotFoundError(f"Config file not found at {config_path}")

        dia = cls(config, compute_dtype, device, load_dac)

        try:
            state_dict = torch.load(checkpoint_path, map_location=dia.device)
            dia.model.load_state_dict(state_dict)
        except FileNotFoundError:
            raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading checkpoint from {checkpoint_path}") from e

        dia.model.to(dia.device)
        dia.model.eval()
        if load_dac:
            dia._load_dac_model()
        return dia

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "nari-labs/Dia-1.6B-0626",
        compute_dtype: str | ComputeDtype = ComputeDtype.FLOAT32,
        device: torch.device | None = None,
        load_dac: bool = True,
    ) -> "Dia":
        """Loads the Dia model from a Hugging Face Hub repository.

        Downloads the configuration and checkpoint files from the specified
        repository ID and then loads the model.

        Args:
            model_name: The Hugging Face Hub repository ID (e.g., "nari-labs/Dia-1.6B-0626").
            compute_dtype: The computation dtype to use.
            device: The device to load the model onto. If None, will automatically select the best available device.
            load_dac: Whether to load the DAC model.

        Returns:
            An instance of the Dia model loaded with weights and set to eval mode.

        Raises:
            FileNotFoundError: If config or checkpoint download/loading fails.
            RuntimeError: If there is an error loading the checkpoint.
        """
        if isinstance(compute_dtype, str):
            compute_dtype = ComputeDtype(compute_dtype)

        # Load model directly using DiaModel's from_pretrained which handles HF download
        try:
            loaded_model = DiaModel.from_pretrained(model_name, compute_dtype=compute_dtype.to_dtype())
        except Exception as e:
            raise RuntimeError(f"Error loading model from Hugging Face Hub ({model_name})") from e

        config = loaded_model.config  # Get config from the loaded model
        dia = cls(config, compute_dtype, device, load_dac)

        dia.model = loaded_model  # Assign the already loaded model
        dia.model.to(dia.device)
        dia.model.eval()
        if load_dac:
            dia._load_dac_model()
        return dia

    def _load_dac_model(self):
        """Loads the Descript Audio Codec (DAC) model.

        Downloads the DAC model if necessary and loads it onto the specified device.
        Sets the DAC model to evaluation mode.

        Raises:
            RuntimeError: If downloading or loading the DAC model fails.
        """
        import dac

        try:
            dac_model_path = dac.utils.download()
            dac_model = dac.DAC.load(dac_model_path).to(self.device)
            dac_model.eval()  # Ensure DAC is in eval mode
        except Exception as e:
            raise RuntimeError("Failed to load DAC model") from e
        self.dac_model = dac_model

    def _encode_text(self, text: str) -> torch.Tensor:
        """Encodes the input text string into a tensor of token IDs using byte-level encoding.

        Special tokens [S1] and [S2] are replaced by their byte values. The resulting
        sequence is truncated to the maximum configured text length.

        Args:
            text: The input text string.

        Returns:
            A tensor containing the encoded byte token IDs.
        """
        max_len = self.config.encoder_config.max_position_embeddings

        byte_text = text.encode("utf-8")
        # Replace special tokens with their byte values if needed by the specific tokenizer/config
        # Assuming byte values 1 and 2 are correct placeholders based on original code
        replaced_bytes = byte_text.replace(b"[S1]", b"\x01").replace(b"[S2]", b"\x02")
        text_tokens = list(replaced_bytes)
        return torch.tensor(
            text_tokens[:max_len],
            dtype=torch.long,
            device=self.device,
        )

    def _pad_text_input(self, text_tokens: list[torch.Tensor]) -> torch.Tensor:
        """Pads the text input to the maximum length."""
        text_pad_value = 0
        max_len = self.config.encoder_config.max_position_embeddings
        batch_size = len(text_tokens)

        src_tokens = torch.full(
            (batch_size, 1, max_len),
            fill_value=text_pad_value,
            dtype=torch.long,
            device=self.device,
        )
        for i in range(batch_size):
            current_len = len(text_tokens[i])
            src_tokens[i, 0, :current_len] = text_tokens[i]
        return src_tokens

    def _prepare_audio_prompt(self, audio_prompts: list[torch.Tensor | None]) -> tuple[torch.Tensor, list[int]]:
        """Prepares the audio prompt tensor for the decoder.

        Handles padding, adds the beginning-of-sequence (BOS) token, applies the
        delay pattern, and determines the number of prefill steps for each item
        in the batch.

        Args:
            audio_prompts: A list of audio prompt tensors (encoded DAC frames) or None.
                           Each tensor should have shape [T, C].

        Returns:
            A tuple containing:
                - delayed_batch (torch.Tensor): The prepared audio prompt tensor with
                  delays applied, shape [B, T_max_padded, C].
                - prefill_steps (list[int]): A list containing the number of valid
                  tokens (including BOS) for each prompt in the batch.
        """
        num_channels = self.config.decoder_config.num_channels
        audio_bos_value = self.config.bos_token_id
        delay_pattern = self.config.delay_pattern
        max_delay_pattern = max(delay_pattern)
        batch_size = len(audio_prompts)

        max_len = max(p.shape[0] if p is not None else 0 for p in audio_prompts) + max_delay_pattern
        prefill_steps = []

        prefill = torch.full(
            (batch_size, max_len, num_channels),
            fill_value=-1,
            dtype=torch.int,
            device=self.device,
        )

        prefill[:, 0, :] = audio_bos_value

        for i in range(batch_size):
            prompt = audio_prompts[i]
            if prompt is not None:
                prompt = prompt.to(device=self.device, dtype=torch.int)
                prefill[i, 1 : prompt.shape[0] + 1, :] = prompt
                prefill_steps.append(prompt.shape[0] + 1)
            else:
                prefill_steps.append(1)

        delay_precomp = build_delay_indices(
            B=batch_size,
            T=max_len,
            C=num_channels,
            delay_pattern=delay_pattern,
        )

        delayed_batch = apply_audio_delay(
            audio_BxTxC=prefill,
            pad_value=-1,
            bos_value=audio_bos_value,
            precomp=delay_precomp,
        )

        return delayed_batch, prefill_steps

    def _prepare_generation(
        self,
        text: torch.Tensor,
        audio_prompts: list[torch.Tensor | None],
        max_tokens: int | None = None,
        attn_fn: Callable = F.scaled_dot_product_attention,
    ):
        """Initializes the model state for generation.

        Encodes the text input (conditional and unconditional), prepares the
        encoder and decoder states (including KV caches and cross-attention),
        prepares the audio prompt, and performs the initial decoder prefill steps
        based on the audio prompts.

        Args:
            text: The padded text input tensor, shape [B, 1, T_text].
            audio_prompts: A list of prepared audio prompt tensors or None.

        Returns:
            A tuple containing:
                - dec_state (DecoderInferenceState): The initialized decoder state.
                - dec_output (DecoderOutput): The initialized decoder output manager,
                  containing the prefilled audio tokens.
        """
        batch_size = text.shape[0]

        enc_input_uncond = torch.zeros_like(text)
        enc_input_cond = text
        stacked_inputs = torch.stack([enc_input_uncond, enc_input_cond], dim=1)
        enc_input = stacked_inputs.view(2 * batch_size, -1)

        enc_state = EncoderInferenceState.new(self.config, enc_input_cond)
        encoder_out = self.model.encoder(enc_input, enc_state)

        dec_cross_attn_cache = self.model.decoder.precompute_cross_attn_cache(encoder_out)
        dec_state = DecoderInferenceState.new(
            self.config,
            enc_state,
            encoder_out,
            dec_cross_attn_cache,
            self.compute_dtype,
            max_generation_length=max_tokens,
        )
        prefill, prefill_steps = self._prepare_audio_prompt(audio_prompts)

        dec_output = DecoderOutput.new(batch_size, self.config, self.device)
        dec_output.prefill(prefill, prefill_steps)

        dec_step = min(prefill_steps) - 1
        if dec_step > 0:
            dec_state.prepare_step(0, dec_step)
            tokens_BxTxC = dec_output.get_tokens_at(0, dec_step).repeat_interleave(2, dim=0)
            self.model.decoder.forward(tokens_BxTxC, dec_state)

        return dec_state, dec_output

    def _decoder_step(
        self,
        tokens_Bx1xC: torch.Tensor,
        dec_state: DecoderInferenceState,
        cfg_scale: float,
        temperature: float,
        top_p: float,
        top_k: int,
        current_idx: int,
    ) -> torch.Tensor:
        """Performs a single step of the decoder inference.

        Takes the tokens from the previous step, runs them through the decoder
        (for both conditional and unconditional paths), applies classifier-free
        guidance (CFG), samples the next token using temperature, top-p, and top-k
        sampling, and applies constraints (e.g., preventing EOS in certain channels).

        Args:
            tokens_Bx1xC: The input tokens for the current step, shape [2*B, 1, C].
                         Repeated for CFG (unconditional and conditional).
            dec_state: The current state of the decoder (KV caches, etc.).
            cfg_scale: The scale factor for classifier-free guidance.
            temperature: The temperature for sampling.
            top_p: The cumulative probability threshold for top-p sampling.
            top_k: The number of top logits to consider for top-k sampling.
            current_idx: The current generation step index.

        Returns:
            torch.Tensor: The sampled next tokens for each item in the batch,
                          shape [B, C].
        """
        B = tokens_Bx1xC.shape[0] // 2

        audio_eos_value = self.config.eos_token_id
        logits_Bx1xCxV = self.model.decoder.decode_step(tokens_Bx1xC, dec_state, current_idx)

        logits_last_2BxCxV = logits_Bx1xCxV[:, -1]
        logits_last_Bx2xCxV = logits_last_2BxCxV.view(B, 2, *logits_last_2BxCxV.shape[1:])

        uncond_logits_BxCxV = logits_last_Bx2xCxV[:, 0, :, :]  # Shape [B, C, V]
        cond_logits_BxCxV = logits_last_Bx2xCxV[:, 1, :, :]  # Shape [B, C, V]
        logits_BxCxV = cond_logits_BxCxV + cfg_scale * (cond_logits_BxCxV - uncond_logits_BxCxV)

        _, top_k_indices_BxCxk = torch.topk(logits_BxCxV, k=top_k, dim=-1)
        mask_BxCxV = torch.ones_like(logits_BxCxV, dtype=torch.bool)
        mask_BxCxV = mask_BxCxV.scatter(dim=-1, index=top_k_indices_BxCxk, value=False)
        logits_BxCxV = cond_logits_BxCxV.masked_fill(mask_BxCxV, -torch.inf)

        logits_BxCxV[:, :, audio_eos_value + 1 :] = torch.full_like(
            logits_BxCxV[:, :, audio_eos_value + 1 :],
            fill_value=-torch.inf,
        )
        logits_BxCxV[:, 1:, audio_eos_value:] = torch.full_like(
            logits_BxCxV[:, 1:, audio_eos_value:],
            fill_value=-torch.inf,
        )

        flat_logits_BCxV = logits_BxCxV.view(B * self.config.decoder_config.num_channels, -1)

        pred_BC = _sample_next_token(
            flat_logits_BCxV.float(),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            audio_eos_value=audio_eos_value,
        )

        pred_BxC = pred_BC.view(B, self.config.decoder_config.num_channels)
        return pred_BxC

    def _generate_output(self, generated_codes: torch.Tensor, lengths_Bx: torch.Tensor) -> list[np.ndarray]:
        """Converts generated delayed codes into audio waveforms.

        Reverts the delay pattern applied during generation, decodes the resulting
        codebook using the DAC model (if loaded), and returns a list of audio
        waveforms as NumPy arrays. If DAC is not loaded, returns the raw codebook indices.

        Args:
            generated_codes: The tensor of generated audio codes with delays,
                             shape [B, T_gen, C].
            lengths_Bx: A tensor containing the valid length of generated codes
                        (excluding padding and BOS/EOS markers) for each item
                        in the batch, shape [B].

        Returns:
            A list of NumPy arrays, where each array represents the generated audio
            waveform for one item in the batch. If DAC is not loaded, returns the
            raw, reverted codebook indices as NumPy arrays.
        """
        num_channels = self.config.decoder_config.num_channels
        batch_size = generated_codes.shape[0]
        seq_length = generated_codes.shape[1]
        delay_pattern = self.config.delay_pattern
        audio_pad_value = self.config.pad_token_id
        max_delay_pattern = max(delay_pattern)

        revert_precomp = build_revert_indices(
            B=batch_size,
            T=seq_length,
            C=num_channels,
            delay_pattern=delay_pattern,
        )

        codebook = revert_audio_delay(
            audio_BxTxC=generated_codes,
            pad_value=audio_pad_value,
            precomp=revert_precomp,
            T=seq_length,
        )[:, :-max_delay_pattern, :]

        min_valid_index = 0
        max_valid_index = 1023
        invalid_mask = (codebook < min_valid_index) | (codebook > max_valid_index)
        codebook[invalid_mask] = 0

        audios = []

        if self.load_dac:
            for i in range(batch_size):
                audio = self._decode(codebook[i, : lengths_Bx[i], :])
                audio_np = audio.cpu().numpy()
                audios.append(audio_np)
        else:
            for i in range(batch_size):
                audios.append(codebook[i, : lengths_Bx[i], :].cpu().numpy())
        return audios

    @torch.no_grad()
    @torch.inference_mode()
    def _encode(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Encodes the given audio waveform into a tensor of DAC codebook indices
        """
        audio = audio.unsqueeze(0)
        audio_data = self.dac_model.preprocess(audio, DEFAULT_SAMPLE_RATE)
        _, encoded_frame, _, _, _ = self.dac_model.encode(audio_data)
        encoded_frame: torch.Tensor
        return encoded_frame.squeeze(0).transpose(0, 1)

    @torch.no_grad()
    @torch.inference_mode()
    def _decode(self, audio_codes: torch.Tensor) -> torch.Tensor:
        """
        Decodes the given frames into an output audio waveform
        """
        audio_codes = audio_codes.unsqueeze(0).transpose(1, 2)
        audio_values, _, _ = self.dac_model.quantizer.from_codes(audio_codes)
        audio_values = self.dac_model.decode(audio_values)
        audio_values: torch.Tensor
        return audio_values.squeeze()

    def load_audio(self, audio_path: str) -> torch.Tensor:
        """Loads and preprocesses an audio file for use as a prompt.

        Loads the audio file, resamples it to the target sample rate if necessary,
        preprocesses it using the DAC model's preprocessing, and encodes it into
        DAC codebook indices.

        Args:
            audio_path: Path to the audio file.

        Returns:
            torch.Tensor: The encoded audio prompt as DAC codebook indices,
                          shape [T, C].

        Raises:
            RuntimeError: If the DAC model is not loaded (`load_dac=False` during init).
            FileNotFoundError: If the audio file cannot be found.
            Exception: If there's an error during loading or processing.
        """
        if self.dac_model is None:
            raise RuntimeError("DAC model is required for loading audio prompts but was not loaded.")
        audio, sr = torchaudio.load(audio_path, channels_first=True)  # C, T
        if sr != DEFAULT_SAMPLE_RATE:
            audio = torchaudio.functional.resample(audio, sr, DEFAULT_SAMPLE_RATE)
        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)  # Average channels to get mono
        return self._encode(audio.to(self.device))

    def save_audio(self, path: str, audio: np.ndarray):
        """Saves the generated audio waveform to a file.

        Uses the soundfile library to write the NumPy audio array to the specified
        path with the default sample rate.

        Args:
            path: The path where the audio file will be saved.
            audio: The audio waveform as a NumPy array.
        """
        import soundfile as sf

        sf.write(path, audio, DEFAULT_SAMPLE_RATE)

    @torch.inference_mode()
    def generate_streaming(
        self,
        text: str | list[str],
        max_tokens: int = 3072,
        cfg_scale: float = 3.0,
        temperature: float = 1.2,
        top_p: float = 0.95,
        use_torch_compile: bool = False,
        cfg_filter_top_k: int = 45,
        audio_prompt: list[str | torch.Tensor | None] | str | torch.Tensor | None = None,
        audio_prompt_path: list[str | torch.Tensor | None] | str | torch.Tensor | None = None,
        chunk_size: int = 512,
        verbose: bool = False,
    ) -> Generator[np.ndarray | list[np.ndarray], None, None]:
        """Generates audio from a text prompt in a streaming fashion.

        This method produces audio in chunks, making it suitable for real-time
        applications. It processes the entire context through the vocoder for
        better quality, then yields only the new audio portion.

        Args:
            text: The input text prompt to synthesize.
            max_tokens: The maximum number of audio tokens to generate.
            cfg_scale: The scale for Classifier-Free Guidance (CFG), controlling
                       how strictly the output adheres to the prompt.
            temperature: The sampling temperature, controlling the randomness of the
                         output. Higher values lead to more varied audio.
            top_p: The nucleus sampling probability, controlling the diversity of
                   the generated tokens.
            use_torch_compile: If True, compiles the decoder step for faster inference.
            cfg_filter_top_k: The number of top-k tokens to consider for CFG filtering.
            audio_prompt: An optional audio prompt (file path or tensor) to condition
                          the generation on a specific voice or style.
            chunk_size: The number of new audio tokens to generate before yielding
                        an audio chunk.
            overlap: The overlap parameter is kept for compatibility but the approach
                     now uses full context processing instead of token-level overlap.
            verbose: If True, prints progress and timing information to the console.

        Yields:
            np.ndarray: A chunk of the generated audio waveform as a NumPy array.
        """
        batch_size = len(text) if isinstance(text, list) else 1
        print(f"batch_size: {batch_size}")
        audio_eos_value = self.config.eos_token_id
        audio_pad_value = self.config.pad_token_id
        delay_pattern = self.config.delay_pattern
        max_delay_pattern = max(delay_pattern)
        delay_pattern_Cx = torch.tensor(delay_pattern, device=self.device, dtype=torch.long)
        self.model.eval()

        if audio_prompt_path:
            audio_prompt = audio_prompt_path

        if use_torch_compile and not hasattr(self, "_compiled"):
            self._prepare_generation = torch.compile(self._prepare_generation, dynamic=True, fullgraph=True)
            self._decoder_step = torch.compile(self._decoder_step, fullgraph=True, mode="max-autotune")
            self._compiled = True

        if isinstance(audio_prompt, list):
            audio_prompt = [self.load_audio(p) if isinstance(p, str) else p for p in audio_prompt]
        elif isinstance(audio_prompt, str):
            audio_prompt = [self.load_audio(audio_prompt)]
        elif isinstance(audio_prompt, torch.Tensor):
            audio_prompt = [audio_prompt]
        elif audio_prompt is None:
            audio_prompt = [None] * batch_size

        assert isinstance(audio_prompt, list), "audio_prompt must be a list of tensors or None"
        assert len(audio_prompt) == batch_size, "Number of audio prompts must match batch size"

        if isinstance(text, list):
            text_tokens = [self._encode_text(t) for t in text]
        else:
            text_tokens = [self._encode_text(text)]
        text_padded = self._pad_text_input(text_tokens)

        dec_state, dec_output = self._prepare_generation(text_padded, audio_prompt, max_tokens=max_tokens)
        dec_step = min(dec_output.prefill_steps) - 1
        current_idx = torch.tensor([dec_step], device=self.device)

        eos_detected_Bx = torch.zeros((batch_size,), dtype=torch.bool, device=self.device)
        eos_countdown_Bx = torch.full((batch_size,), -1, dtype=torch.long, device=self.device)
        finished_step_Bx = torch.full((batch_size,), -1, dtype=torch.long, device=self.device)

        bos_over = False
        if verbose:
            print("generate: starting generation loop")
            if use_torch_compile:
                print("generate: using use_torch_compile=True, the first step may be slow")
            start_time = time.time()
        audio_samples_yielded = [0] * batch_size
        last_yield_step = dec_step

        while dec_step < max_tokens:
            if (eos_countdown_Bx == 0).all():
                break

            current_step_idx = dec_step + 1
            torch.compiler.cudagraph_mark_step_begin()
            dec_state.prepare_step(dec_step)
            tokens_Bx1xC = dec_output.get_tokens_at(dec_step).repeat_interleave(2, dim=0)

            pred_BxC = self._decoder_step(
                tokens_Bx1xC,
                dec_state,
                cfg_scale,
                temperature,
                top_p,
                cfg_filter_top_k,
                current_idx,
            )
            current_idx += 1

            active_mask_Bx = eos_countdown_Bx != 0
            eos_trigger_Bx = torch.zeros_like(active_mask_Bx)
            if active_mask_Bx.any():
                is_eos_token = (~eos_detected_Bx[active_mask_Bx]) & (pred_BxC[active_mask_Bx, 0] == audio_eos_value)
                is_max_len = current_step_idx >= max_tokens - max_delay_pattern
                eos_trigger_Bx[active_mask_Bx] = is_eos_token | is_max_len
            eos_detected_Bx |= eos_trigger_Bx
            start_countdown_mask_Bx = eos_trigger_Bx & (eos_countdown_Bx < 0)
            if start_countdown_mask_Bx.any():
                eos_countdown_Bx[start_countdown_mask_Bx] = max_delay_pattern
                finished_step_Bx[start_countdown_mask_Bx] = current_step_idx

            padding_mask_Bx = eos_countdown_Bx > 0
            if padding_mask_Bx.any():
                pred_active_BxC = pred_BxC[padding_mask_Bx].clone()
                countdown_active_Bx = eos_countdown_Bx[padding_mask_Bx]
                step_after_eos_Bx = max_delay_pattern - countdown_active_Bx
                step_after_eos_Bx_ = step_after_eos_Bx.unsqueeze(1)
                delay_pattern_Cx_ = delay_pattern_Cx.unsqueeze(0)
                eos_mask_NxC = step_after_eos_Bx_ == delay_pattern_Cx_
                pad_mask_NxC = step_after_eos_Bx_ > delay_pattern_Cx_
                pred_active_BxC[eos_mask_NxC] = audio_eos_value
                pred_active_BxC[pad_mask_NxC] = audio_pad_value
                pred_BxC[padding_mask_Bx] = pred_active_BxC
                eos_countdown_Bx[padding_mask_Bx] -= 1

            if not bos_over:
                bos_over = all(
                    dec_step - prefill_step > max_delay_pattern for prefill_step in dec_output.prefill_steps
                )

            dec_output.update_one(pred_BxC, current_step_idx, not bos_over)

            dec_step += 1

            if verbose and dec_step % 86 == 0:
                duration = time.time() - start_time
                if duration > 0:
                    print(
                        f"generate step {dec_step}: speed={86 * batch_size / duration:.3f} tokens/s, realtime factor={batch_size / duration:.3f}x"
                    )
                start_time = time.time()

            if current_step_idx - last_yield_step >= chunk_size:
                prefill_start = dec_output.prefill_steps
                all_tokens = [
                    dec_output.generated_tokens[i, prefill_start[i] : current_step_idx, :] for i in range(batch_size)
                ]
                total_lens = [tokens.shape[0] for tokens in all_tokens]
                num_channels = self.config.decoder_config.num_channels
                generated_codes = torch.full(
                    (batch_size, max(total_lens), num_channels),
                    fill_value=audio_pad_value,
                    dtype=torch.long,
                    device=self.device,
                )
                for i in range(batch_size):
                    generated_codes[i, : total_lens[i], :] = all_tokens[i]
                lengths_Bx = torch.tensor(total_lens, device=self.device)
                audio_chunks = self._generate_output(generated_codes, lengths_Bx)
                new_audios = []
                for i, full_audio in enumerate(audio_chunks):
                    if full_audio.size > 0:
                        new_audio = full_audio[audio_samples_yielded[i] :]
                        if new_audio.size > 0:
                            new_audios.append(new_audio)
                            audio_samples_yielded[i] = len(full_audio)
                        else:
                            new_audios.append(np.array([]))
                    else:
                        new_audios.append(np.array([]))
                yield new_audios[0] if batch_size == 1 else new_audios
                last_yield_step = current_step_idx

        if current_step_idx > last_yield_step:
            prefill_start = dec_output.prefill_steps
            all_tokens = [
                dec_output.generated_tokens[i, prefill_start[i] : current_step_idx, :] for i in range(batch_size)
            ]
            total_lens = [tokens.shape[0] for tokens in all_tokens]
            num_channels = self.config.decoder_config.num_channels
            generated_codes = torch.full(
                (batch_size, max(total_lens), num_channels),
                fill_value=audio_pad_value,
                dtype=torch.long,
                device=self.device,
            )
            for i in range(batch_size):
                generated_codes[i, : total_lens[i], :] = all_tokens[i]
            lengths_Bx = torch.tensor(total_lens, device=self.device)
            audio_chunks = self._generate_output(generated_codes, lengths_Bx)
            new_audios = []
            for i, full_audio in enumerate(audio_chunks):
                if full_audio.size > 0:
                    new_audio = full_audio[audio_samples_yielded[i] :]
                    if new_audio.size > 0:
                        new_audios.append(new_audio)
                    else:
                        new_audios.append(np.array([]))
                else:
                    new_audios.append(np.array([]))
            yield new_audios[0] if batch_size == 1 else new_audios

    @torch.inference_mode()
    def generate(
        self,
        text: str | list[str],
        max_tokens: int = 3072,
        cfg_scale: float = 3.0,
        temperature: float = 1.2,
        top_p: float = 0.95,
        use_torch_compile: bool = False,
        cfg_filter_top_k: int = 45,
        audio_prompt: list[str | torch.Tensor | None] | str | torch.Tensor | None = None,
        audio_prompt_path: list[str | torch.Tensor | None] | str | torch.Tensor | None = None,
        use_cfg_filter: bool | None = None,
        verbose: bool = False,
    ) -> np.ndarray | list[np.ndarray]:
        """Generates audio corresponding to the input text.

        Args:
            text: The input text prompt, or a list of text prompts for batch generation.
            max_tokens: The maximum number of audio tokens to generate per prompt.
                        Defaults to the model's configured audio length if None.
            cfg_scale: The scale factor for classifier-free guidance (CFG). Higher values
                       lead to stronger guidance towards the text prompt.
            temperature: The temperature for sampling. Higher values increase randomness.
            top_p: The cumulative probability threshold for nucleus (top-p) sampling.
            use_torch_compile: Whether to compile the generation steps using torch.compile.
                               Can significantly speed up generation after the initial
                               compilation overhead. Defaults to False.
            cfg_filter_top_k: The number of top logits to consider during CFG filtering.
                              (Note: This parameter name might be slightly misleading based
                              on the code; it's used in the `_sample_next_token` function.)
            audio_prompt: An audio prompt or list of prompts to condition the generation.
                          Can be a file path (str), a pre-loaded tensor (DAC codes), or None.
                          If a list, its length must match the batch size of the text input.
            audio_prompt_path: (Deprecated) Use `audio_prompt` instead.
            use_cfg_filter: (Deprecated) This parameter is no longer used.
            verbose: If True, prints progress information during generation, including
                     speed metrics.

        Returns:
            If a single text prompt was provided, returns a NumPy array containing the
            generated audio waveform.
            If a list of text prompts was provided, returns a list of NumPy arrays,
            each corresponding to a prompt in the input list. Returns None for a
            sequence if no audio was generated for it.
        """
        batch_size = len(text) if isinstance(text, list) else 1
        audio_eos_value = self.config.eos_token_id
        audio_pad_value = self.config.pad_token_id
        delay_pattern = self.config.delay_pattern
        max_delay_pattern = max(delay_pattern)
        delay_pattern_Cx = torch.tensor(delay_pattern, device=self.device, dtype=torch.long)
        self.model.eval()

        if audio_prompt_path:
            print("Warning: audio_prompt_path is deprecated. Use audio_prompt instead.")
            audio_prompt = audio_prompt_path
        if use_cfg_filter is not None:
            print("Warning: use_cfg_filter is deprecated.")

        if verbose:
            total_start_time = time.time()

        if use_torch_compile and not hasattr(self, "_compiled"):
            # Compilation can take about a minute.
            self._prepare_generation = torch.compile(self._prepare_generation, dynamic=True, fullgraph=True)
            self._decoder_step = torch.compile(self._decoder_step, fullgraph=True, mode="max-autotune")
            self._compiled = True

        if isinstance(audio_prompt, list):
            audio_prompt = [self.load_audio(p) if isinstance(p, str) else p for p in audio_prompt]
        elif isinstance(audio_prompt, str):
            audio_prompt = [self.load_audio(audio_prompt)]
        elif isinstance(audio_prompt, torch.Tensor):
            audio_prompt = [audio_prompt]
        elif audio_prompt is None:
            audio_prompt = [None] * batch_size

        assert len(audio_prompt) == batch_size, "Number of audio prompts must match batch size"

        if isinstance(text, list):
            text = [self._encode_text(t) for t in text]
        else:
            text = [self._encode_text(text)]
        text = self._pad_text_input(text)

        dec_state, dec_output = self._prepare_generation(text, audio_prompt, max_tokens=max_tokens)
        dec_step = min(dec_output.prefill_steps) - 1
        current_idx = torch.tensor([dec_step], device=self.device)

        eos_detected_Bx = torch.zeros((batch_size,), dtype=torch.bool, device=self.device)
        eos_countdown_Bx = torch.full((batch_size,), -1, dtype=torch.long, device=self.device)
        finished_step_Bx = torch.full((batch_size,), -1, dtype=torch.long, device=self.device)

        bos_over = False

        if verbose:
            print("generate: starting generation loop")
            if use_torch_compile:
                print("generate: using use_torch_compile=True, the first step may be slow")
            start_time = time.time()

        # --- Generation Loop ---
        while dec_step < max_tokens:
            if (eos_countdown_Bx == 0).all():
                break

            current_step_idx = dec_step + 1
            torch.compiler.cudagraph_mark_step_begin()
            dec_state.prepare_step(dec_step)
            tokens_Bx1xC = dec_output.get_tokens_at(dec_step).repeat_interleave(2, dim=0)  # Repeat for CFG

            pred_BxC = self._decoder_step(
                tokens_Bx1xC,
                dec_state,
                cfg_scale,
                temperature,
                top_p,
                cfg_filter_top_k,
                current_idx,
            )

            current_idx += 1

            active_mask_Bx = eos_countdown_Bx != 0
            eos_trigger_Bx = torch.zeros_like(active_mask_Bx)
            if active_mask_Bx.any():
                is_eos_token = (~eos_detected_Bx[active_mask_Bx]) & (pred_BxC[active_mask_Bx, 0] == audio_eos_value)
                is_max_len = current_step_idx >= max_tokens - max_delay_pattern
                eos_trigger_Bx[active_mask_Bx] = is_eos_token | is_max_len
            eos_detected_Bx |= eos_trigger_Bx
            start_countdown_mask_Bx = eos_trigger_Bx & (eos_countdown_Bx < 0)
            if start_countdown_mask_Bx.any():
                eos_countdown_Bx[start_countdown_mask_Bx] = max_delay_pattern
                finished_step_Bx[start_countdown_mask_Bx] = current_step_idx

            padding_mask_Bx = eos_countdown_Bx > 0
            if padding_mask_Bx.any():
                pred_active_BxC = pred_BxC[padding_mask_Bx].clone()
                countdown_active_Bx = eos_countdown_Bx[padding_mask_Bx]
                step_after_eos_Bx = max_delay_pattern - countdown_active_Bx
                step_after_eos_Bx_ = step_after_eos_Bx.unsqueeze(1)
                delay_pattern_Cx_ = delay_pattern_Cx.unsqueeze(0)
                eos_mask_NxC = step_after_eos_Bx_ == delay_pattern_Cx_
                pad_mask_NxC = step_after_eos_Bx_ > delay_pattern_Cx_
                pred_active_BxC[eos_mask_NxC] = audio_eos_value
                pred_active_BxC[pad_mask_NxC] = audio_pad_value
                pred_BxC[padding_mask_Bx] = pred_active_BxC
                eos_countdown_Bx[padding_mask_Bx] -= 1

            # --- Update BOS flag (Original) ---
            if not bos_over:
                bos_over = all(
                    dec_step - prefill_step > max_delay_pattern for prefill_step in dec_output.prefill_steps
                )

            dec_output.update_one(pred_BxC, current_step_idx, not bos_over)

            dec_step += 1

            if verbose and dec_step % 86 == 0:
                duration = time.time() - start_time
                if duration > 0:
                    print(
                        f"generate step {dec_step}: speed={86 * batch_size / duration:.3f} tokens/s, realtime factor={batch_size / duration:.3f}x"
                    )
                start_time = time.time()

        # --- Finalize and Extract Output ---
        final_step = dec_step + 1

        finished_step_Bx[finished_step_Bx == -1] = final_step - max_delay_pattern

        prefill_steps_tensor = torch.tensor(dec_output.prefill_steps, device=self.device)
        lengths_Bx = finished_step_Bx - prefill_steps_tensor
        lengths_Bx = torch.clamp(lengths_Bx, min=0)

        max_len = lengths_Bx.max().item() + max_delay_pattern
        outputs = []

        if max_len > 0:
            num_channels = self.config.decoder_config.num_channels
            audio_pad_value = self.config.pad_token_id
            generated_codes = torch.full(
                (batch_size, max_len, num_channels),
                fill_value=audio_pad_value,
                dtype=torch.long,
                device=self.device,
            )

            for i in range(batch_size):
                start_step = dec_output.prefill_steps[i]
                actual_len = lengths_Bx[i].item() + max_delay_pattern
                if actual_len > 0:
                    tokens_to_copy = dec_output.generated_tokens[i, start_step : start_step + actual_len, :]
                    generated_codes[i, :actual_len, :] = tokens_to_copy

            if verbose:
                avg_steps = lengths_Bx.float().mean().item()
                total_duration = time.time() - total_start_time
                print(f"generate: avg steps={avg_steps:.1f}, total duration={total_duration:.3f}s")

            del dec_state

            outputs = self._generate_output(generated_codes, lengths_Bx)
        else:
            print("Warning: Nothing generated for any sequence in the batch.")
            outputs = [None] * batch_size

        return outputs if batch_size > 1 else outputs[0]
