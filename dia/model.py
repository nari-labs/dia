import time
from enum import Enum

import dac
import numpy as np
import torch
import torchaudio

from .audio import apply_audio_delay, build_delay_indices, build_revert_indices, decode, revert_audio_delay
from .config import DiaConfig
from .layers import DiaModel
from .state import DecoderInferenceState, DecoderOutput, EncoderInferenceState


DEFAULT_SAMPLE_RATE = 44100


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
    cfg_filter_top_k: int | None = None,
) -> torch.Tensor:
    if temperature == 0.0:
        return torch.argmax(logits_BCxV, dim=-1)

    logits_BCxV = logits_BCxV / temperature
    if cfg_filter_top_k is not None:
        _, top_k_indices_BCxV = torch.topk(logits_BCxV, k=cfg_filter_top_k, dim=-1)
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
    ):
        """Initializes the Dia model.

        Args:
            config: The configuration object for the model.
            device: The device to load the model onto. If None, will automatically select the best available device.

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

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True

    @classmethod
    def from_local(
        cls,
        config_path: str,
        checkpoint_path: str,
        compute_dtype: str | ComputeDtype = ComputeDtype.FLOAT32,
        device: torch.device | None = None,
    ) -> "Dia":
        """Loads the Dia model from local configuration and checkpoint files.

        Args:
            config_path: Path to the configuration JSON file.
            checkpoint_path: Path to the model checkpoint (.pth) file.
            device: The device to load the model onto. If None, will automatically select the best available device.

        Returns:
            An instance of the Dia model loaded with weights and set to eval mode.

        Raises:
            FileNotFoundError: If the config or checkpoint file is not found.
            RuntimeError: If there is an error loading the checkpoint.
        """
        config = DiaConfig.load(config_path)
        if config is None:
            raise FileNotFoundError(f"Config file not found at {config_path}")

        dia = cls(config, compute_dtype, device)

        try:
            state_dict = torch.load(checkpoint_path, map_location=dia.device)
            dia.model.load_state_dict(state_dict)
        except FileNotFoundError:
            raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading checkpoint from {checkpoint_path}") from e

        dia.model.to(dia.device)
        dia.model.eval()
        dia._load_dac_model()
        return dia

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "nari-labs/Dia-1.6B",
        compute_dtype: str | ComputeDtype = ComputeDtype.FLOAT32,
        device: torch.device | None = None,
    ) -> "Dia":
        """Loads the Dia model from a Hugging Face Hub repository.

        Downloads the configuration and checkpoint files from the specified
        repository ID and then loads the model.

        Args:
            model_name: The Hugging Face Hub repository ID (e.g., "nari-labs/Dia-1.6B").
            compute_dtype: The computation dtype to use.
            device: The device to load the model onto. If None, will automatically select the best available device.

        Returns:
            An instance of the Dia model loaded with weights and set to eval mode.

        Raises:
            FileNotFoundError: If config or checkpoint download/loading fails.
            RuntimeError: If there is an error loading the checkpoint.
        """
        if isinstance(compute_dtype, str):
            compute_dtype = ComputeDtype(compute_dtype)
        loaded_model = DiaModel.from_pretrained(model_name, compute_dtype=compute_dtype.to_dtype())
        config = loaded_model.config
        dia = cls(config, compute_dtype, device)

        dia.model = loaded_model
        dia.model.to(dia.device)
        dia.model.eval()
        dia._load_dac_model()
        return dia

    def _load_dac_model(self):
        try:
            dac_model_path = dac.utils.download()
            dac_model = dac.DAC.load(dac_model_path).to(self.device)
        except Exception as e:
            raise RuntimeError("Failed to load DAC model") from e
        self.dac_model = dac_model

    def _prepare_text_input(self, text: str) -> torch.Tensor:
        """Encodes text prompt, pads, and creates attention mask and positions."""
        text_pad_value = self.config.data.text_pad_value
        max_len = self.config.data.text_length

        byte_text = text.encode("utf-8")
        replaced_bytes = byte_text.replace(b"[S1]", b"\x01").replace(b"[S2]", b"\x02")
        text_tokens = list(replaced_bytes)

        current_len = len(text_tokens)
        src_tokens = torch.full(
            (1, max_len),
            fill_value=text_pad_value,
            dtype=torch.long,
            device=self.device,
        )
        src_tokens[0, :current_len] = torch.tensor(
            text_tokens[:max_len],
            dtype=torch.long,
            device=self.device,
        )
        return src_tokens

    def _prepare_audio_prompt(self, audio_prompt: torch.Tensor | None) -> tuple[torch.Tensor, int]:
        num_channels = self.config.data.channels
        audio_bos_value = self.config.data.audio_bos_value
        audio_pad_value = self.config.data.audio_pad_value
        delay_pattern = self.config.data.delay_pattern
        max_delay_pattern = max(delay_pattern)

        parts = [
            torch.full(
                (1, num_channels),
                fill_value=audio_bos_value,
                dtype=torch.int,
                device=self.device,
            )
        ]

        prefill_step = 1

        if audio_prompt is not None:
            prefill_step += audio_prompt.shape[0]
            parts.append(audio_prompt)

        delay_pad_tensor = torch.full(
            (max_delay_pattern, num_channels), fill_value=-1, dtype=torch.int, device=self.device
        )
        parts.append(delay_pad_tensor)
        prefill = torch.cat(parts, dim=0)

        delay_precomp = build_delay_indices(
            B=1,
            T=prefill.shape[0],
            C=num_channels,
            delay_pattern=delay_pattern,
        )

        prefill = apply_audio_delay(
            audio_BxTxC=prefill.unsqueeze(0),
            pad_value=audio_pad_value,
            bos_value=audio_bos_value,
            precomp=delay_precomp,
        ).squeeze(0)

        return prefill, prefill_step

    def _prepare_generation(self, text: str, audio_prompt: str | torch.Tensor | None, verbose: bool):
        enc_input_cond = self._prepare_text_input(text)
        enc_input_uncond = torch.zeros_like(enc_input_cond)
        enc_input = torch.cat([enc_input_uncond, enc_input_cond], dim=0)

        if isinstance(audio_prompt, str):
            audio_prompt = self.load_audio(audio_prompt)
        prefill, prefill_step = self._prepare_audio_prompt(audio_prompt)

        if verbose:
            print("generate: data loaded")

        enc_state = EncoderInferenceState.new(self.config, enc_input_cond)
        encoder_out = self.model.encoder(enc_input, enc_state)

        dec_cross_attn_cache = self.model.decoder.precompute_cross_attn_cache(
            encoder_out, enc_state.positions, enc_state.padding_mask
        )
        dec_state = DecoderInferenceState.new(
            self.config, enc_state, encoder_out, dec_cross_attn_cache, self.compute_dtype
        )
        dec_output = DecoderOutput.new(self.config, self.device)
        dec_output.prefill(prefill, prefill_step)

        dec_step = prefill_step - 1
        if dec_step > 0:
            dec_state.prepare_step(0, dec_step)
            tokens_BxTxC = dec_output.get_tokens_at(0, dec_step).unsqueeze(0).expand(2, -1, -1)
            self.model.decoder.forward(tokens_BxTxC, dec_state)

        return dec_state, dec_output

    def _decoder_step(
        self,
        tokens_Bx1xC: torch.Tensor,
        dec_state: DecoderInferenceState,
        cfg_scale: float,
        temperature: float,
        top_p: float,
        cfg_filter_top_k: int,
        current_idx: int,
    ) -> torch.Tensor:
        audio_eos_value = self.config.data.audio_eos_value
        logits_Bx1xCxV = self.model.decoder.decode_step(tokens_Bx1xC, dec_state, current_idx)

        logits_last_BxCxV = logits_Bx1xCxV[:, -1]
        uncond_logits_CxV = logits_last_BxCxV[0]
        cond_logits_CxV = logits_last_BxCxV[1]
        logits_CxV = cond_logits_CxV + cfg_scale * (cond_logits_CxV - uncond_logits_CxV)
        logits_CxV[:, audio_eos_value + 1 :] = torch.full_like(
            logits_CxV[:, audio_eos_value + 1 :],
            fill_value=-torch.inf,
        )
        logits_CxV[1:, audio_eos_value:] = torch.full_like(
            logits_CxV[1:, audio_eos_value:],
            fill_value=-torch.inf,
        )

        pred_C = _sample_next_token(
            logits_CxV.to(dtype=torch.float32),
            temperature=temperature,
            top_p=top_p,
            cfg_filter_top_k=cfg_filter_top_k,
        )
        return pred_C

    def _generate_output(self, generated_codes: torch.Tensor) -> np.ndarray:
        num_channels = self.config.data.channels
        seq_length = generated_codes.shape[0]
        delay_pattern = self.config.data.delay_pattern
        audio_pad_value = self.config.data.audio_pad_value
        max_delay_pattern = max(delay_pattern)

        revert_precomp = build_revert_indices(
            B=1,
            T=seq_length,
            C=num_channels,
            delay_pattern=delay_pattern,
        )

        codebook = revert_audio_delay(
            audio_BxTxC=generated_codes.unsqueeze(0),
            pad_value=audio_pad_value,
            precomp=revert_precomp,
            T=seq_length,
        )[:, :-max_delay_pattern, :]

        min_valid_index = 0
        max_valid_index = 1023
        invalid_mask = (codebook < min_valid_index) | (codebook > max_valid_index)
        codebook[invalid_mask] = 0

        audio = decode(self.dac_model, codebook.transpose(1, 2))

        return audio.squeeze().cpu().numpy()

    def load_audio(self, audio_path: str) -> torch.Tensor:
        audio, sr = torchaudio.load(audio_path, channels_first=True)  # C, T
        if sr != DEFAULT_SAMPLE_RATE:
            audio = torchaudio.functional.resample(audio, sr, DEFAULT_SAMPLE_RATE)
        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True) # Average channels to get mono
        audio = audio.to(self.device).unsqueeze(0)  # 1, C, T
        audio_data = self.dac_model.preprocess(audio, DEFAULT_SAMPLE_RATE)
        _, encoded_frame, _, _, _ = self.dac_model.encode(audio_data)  # 1, C, T
        return encoded_frame.squeeze(0).transpose(0, 1)

    def save_audio(self, path: str, audio: np.ndarray):
        import soundfile as sf

        sf.write(path, audio, DEFAULT_SAMPLE_RATE)

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        max_tokens: int | None = None,
        cfg_scale: float = 3.0,
        temperature: float = 1.3,
        top_p: float = 0.95,
        use_torch_compile: bool = False,
        cfg_filter_top_k: int = 35,
        audio_prompt: str | torch.Tensor | None = None,
        audio_prompt_path: str | None = None,
        use_cfg_filter: bool | None = None,
        verbose: bool = False,
    ) -> np.ndarray:
        audio_eos_value = self.config.data.audio_eos_value
        audio_pad_value = self.config.data.audio_pad_value
        delay_pattern = self.config.data.delay_pattern
        max_tokens = self.config.data.audio_length if max_tokens is None else max_tokens
        max_delay_pattern = max(delay_pattern)
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
            self._prepare_generation = torch.compile(self._prepare_generation, dynamic=True)
            self._decoder_step = torch.compile(self._decoder_step, fullgraph=True, mode="max-autotune")
            self._compiled = True
        dec_state, dec_output = self._prepare_generation(text, audio_prompt, verbose)
        dec_step = dec_output.prefill_step - 1
        current_idx = torch.tensor([dec_step], device=self.device)

        bos_countdown = max_delay_pattern
        eos_detected = False
        eos_countdown = -1

        if verbose:
            print("generate: starting generation loop")
            if use_torch_compile:
                print("generate: by using use_torch_compile=True, the first step would take long")
            start_time = time.time()

        while dec_step < max_tokens:
            torch.compiler.cudagraph_mark_step_begin()
            dec_state.prepare_step(dec_step)
            tokens_Bx1xC = dec_output.get_tokens_at(dec_step).unsqueeze(0).expand(2, -1, -1)
            pred_C = self._decoder_step(
                tokens_Bx1xC,
                dec_state,
                cfg_scale,
                temperature,
                top_p,
                cfg_filter_top_k,
                current_idx,
            )

            current_idx += 1

            if (not eos_detected and pred_C[0] == audio_eos_value) or dec_step == max_tokens - max_delay_pattern - 1:
                eos_detected = True
                eos_countdown = max_delay_pattern

            if eos_countdown > 0:
                step_after_eos = max_delay_pattern - eos_countdown
                for i, d in enumerate(delay_pattern):
                    if step_after_eos == d:
                        pred_C[i] = audio_eos_value
                    elif step_after_eos > d:
                        pred_C[i] = audio_pad_value
                eos_countdown -= 1

            bos_countdown = max(0, bos_countdown - 1)
            dec_output.update_one(pred_C, dec_step + 1, bos_countdown > 0)

            if eos_countdown == 0:
                break

            dec_step += 1
            if verbose and dec_step % 86 == 0:
                duration = time.time() - start_time
                print(
                    f"generate step {dec_step}: speed={86 / duration:.3f} tokens/s, realtime factor={1 / duration:.3f}x"
                )
                start_time = time.time()

        if dec_output.prefill_step >= dec_step + 1:
            print("Warning: Nothing generated")
            return None

        generated_codes = dec_output.generated_tokens[dec_output.prefill_step : dec_step + 1, :]

        if verbose:
            total_step = dec_step + 1 - dec_output.prefill_step
            total_duration = time.time() - total_start_time
            print(f"generate: total step={total_step}, total duration={total_duration:.3f}s")

        return self._generate_output(generated_codes)
