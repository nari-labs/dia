from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import RMSNorm

from .config import DiaConfig
from .state import KVCache


def _normalize_axes(axes: tuple[int, ...], ndim: int) -> tuple[int, ...]:
    return tuple(ax if ax >= 0 else ndim + ax for ax in axes)


def _str_to_dtype(dtype_str: str) -> torch.dtype | None:
    # Allow None for default behavior
    if dtype_str is None or dtype_str.lower() == "none":
        return None
    if dtype_str == "float32":
        return torch.float32
    elif dtype_str == "float16":
        return torch.float16
    elif dtype_str == "bfloat16":
        return torch.bfloat16
    else:
        raise ValueError(f"Unsupported dtype string: {dtype_str}")


class DenseGeneral(nn.Module):
    """
    PyTorch equivalent of flax.linen.DenseGeneral with shapes defined at init.

    Stores weights (`kernel`) in the same layout as Jax and uses torch.tensordot
    for the generalized matrix multiplication. Weight/bias shapes are calculated
    and parameters created during initialization based on config.
    `load_weights` validates shapes and copies data.

    Attributes:
        axis (Tuple[int, ...]): Input axis or axes to contract.
        in_shapes (Tuple[int, ...]): Sizes of the input dimensions specified by `axis`.
        out_features (Tuple[int, ...]): Shape of the output features (non-contracted dims).
        use_bias (bool): Whether to add a bias term.
        weight (nn.Parameter): The kernel parameter.
        bias (Optional[nn.Parameter]): The bias parameter (if use_bias=True).
    """

    def __init__(
        self,
        in_shapes: tuple[int, ...],
        out_features: tuple[int, ...],
        axis: tuple[int, ...] = (-1,),
        weight_dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.in_shapes = in_shapes
        self.out_features = out_features
        self.axis = axis
        self.kernel_shape = self.in_shapes + self.out_features

        factory_kwargs = {"device": device, "dtype": weight_dtype}
        self.weight = nn.Parameter(torch.empty(self.kernel_shape, **factory_kwargs))
        self.register_parameter("bias", None)

    def forward(self, inputs: Tensor) -> Tensor:
        norm_axis = _normalize_axes(self.axis, inputs.ndim)
        kernel_contract_axes = tuple(range(len(norm_axis)))

        output = torch.tensordot(
            inputs.float(),
            self.weight.float(),
            dims=(norm_axis, kernel_contract_axes),
        ).to(inputs.dtype)
        return output


class MlpBlock(nn.Module):
    """MLP block using DenseGeneral."""

    def __init__(
        self,
        config: DiaConfig,
        embed_dim: int,
        intermediate_dim: int,
    ):
        super().__init__()
        compute_dtype = _str_to_dtype(config.training.dtype)
        weight_dtype = _str_to_dtype(config.model.weight_dtype)
        self.dtype = compute_dtype

        self.wi_fused = DenseGeneral(
            in_shapes=(embed_dim,),
            out_features=(2, intermediate_dim),
            axis=(-1,),
            weight_dtype=weight_dtype,
        )

        self.wo = DenseGeneral(
            in_shapes=(intermediate_dim,),
            out_features=(embed_dim,),
            axis=(-1,),
            weight_dtype=weight_dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        fused_x = self.wi_fused(x)

        gate = fused_x[..., 0, :]
        up = fused_x[..., 1, :]

        hidden = torch.mul(F.silu(gate), up).to(self.dtype)

        output = self.wo(hidden)
        return output


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) implementation in PyTorch."""

    def __init__(
        self,
        embedding_dims: int,
        min_timescale: int = 1,
        max_timescale: int = 10000,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if embedding_dims % 2 != 0:
            raise ValueError("Embedding dim must be even for RoPE.")
        self.embedding_dims = embedding_dims
        self.min_timescale = min_timescale
        self.max_timescale = max_timescale
        self.dtype = dtype

        half_embedding_dim = embedding_dims // 2
        fraction = (2.0 * torch.arange(0, half_embedding_dim)) / embedding_dims
        self.register_buffer(
            "timescale",
            self.min_timescale * (self.max_timescale / self.min_timescale) ** fraction,
            persistent=False,
        )

    def extra_repr(self) -> str:
        s = f"{self.timescale.shape}"
        return s

    def forward(self, inputs: torch.Tensor, position: torch.Tensor):
        """Applies RoPE."""
        position = position.unsqueeze(-1).unsqueeze(-1)
        timescale = self.timescale.to(inputs.device)
        sinusoid_inp = position / timescale
        sin = torch.sin(sinusoid_inp).to(inputs.dtype)
        cos = torch.cos(sinusoid_inp).to(inputs.dtype)
        first_half, second_half = torch.chunk(inputs, 2, dim=-1)
        first_part = first_half * cos - second_half * sin
        second_part = second_half * cos + first_half * sin
        return torch.cat((first_part, second_part), dim=-1)


class Attention(nn.Module):
    """Attention using DenseGeneral."""

    def __init__(
        self,
        config: DiaConfig,
        q_embed_dim: int,
        kv_embed_dim: int,
        num_query_heads: int,
        num_kv_heads: int,
        head_dim: int,
        is_cross_attn: bool = False,
        out_embed_dim: int | None = None,
    ):
        super().__init__()
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.is_cross_attn = is_cross_attn
        compute_dtype = _str_to_dtype(config.training.dtype)
        weight_dtype = _str_to_dtype(config.model.weight_dtype)
        self.output_dim = out_embed_dim if out_embed_dim is not None else q_embed_dim
        self.projected_query_dim = num_query_heads * head_dim
        if num_query_heads % num_kv_heads != 0:
            raise ValueError(f"num_query_heads ({num_query_heads}) must be divisible by num_kv_heads ({num_kv_heads})")
        self.num_gqa_groups = num_query_heads // num_kv_heads

        # --- Projection Layers using DenseGeneral ---
        self.q_proj = DenseGeneral(
            in_shapes=(q_embed_dim,),
            out_features=(num_query_heads, head_dim),
            axis=(-1,),
            weight_dtype=weight_dtype,
        )
        self.k_proj = DenseGeneral(
            in_shapes=(kv_embed_dim,),
            out_features=(num_kv_heads, head_dim),
            axis=(-1,),
            weight_dtype=weight_dtype,
        )
        self.v_proj = DenseGeneral(
            in_shapes=(kv_embed_dim,),
            out_features=(num_kv_heads, head_dim),
            axis=(-1,),
            weight_dtype=weight_dtype,
        )
        self.o_proj = DenseGeneral(
            in_shapes=(num_query_heads, head_dim),
            out_features=(self.output_dim,),
            axis=(-2, -1),
            weight_dtype=weight_dtype,
        )

        # --- Rotary Embedding ---
        self.rotary_emb = RotaryEmbedding(
            embedding_dims=self.head_dim,
            min_timescale=config.model.rope_min_timescale,
            max_timescale=config.model.rope_max_timescale,
            dtype=compute_dtype,
        )

    def forward(
        self,
        Xq: torch.Tensor,  # (B, T, D) T = 1 in AR generation
        Xkv: torch.Tensor,  # (B, S, E) S = 1 in AR generation
        q_positions: torch.Tensor,  # (B, T)
        kv_positions: torch.Tensor | None = None,  # (B, S)
        attn_mask: torch.Tensor | None = None,  # None in Decoder Self Attention, Valid mask in Others
        cache: KVCache | None = None,  # None in Encoder, KVCache in Decoder
        prefill: bool = False,  # True only when prefilling KV Cache
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """
        Performs attention calculation with optional KV caching.

        Args:
            Xq: Query tensor (B, T, D). T=1 during single-step decoding.
            Xkv: Key/Value source tensor (B, S, E). S=1 during single-step decoding for self-attn.
            q_positions: Positions for queries (B, T).
            kv_positions: Positions for keys/values (B, S). If None, uses q_positions.
            attn_mask: Attention mask.
            cache: KVCache.
            prefill: If True, use prefill mode.

        Returns:
            A tuple containing:
            - output: The attention output tensor (B, T, output_dim).
            - present_kv: The K/V state to be cached for the next step ((B, N, S_new, H), (B, N, S_new, H)). For self-attn, S_new = S_past + S. For cross-attn, S_new = S_kv.
        """
        if kv_positions is None:
            kv_positions = q_positions
        original_dtype = Xq.dtype

        Xq_BxTxNxH = self.q_proj(Xq)
        Xq_BxTxNxH = self.rotary_emb(Xq_BxTxNxH, position=q_positions)
        Xq_BxNxTxH = Xq_BxTxNxH.transpose(1, 2)

        attn_k: torch.Tensor | None = None
        attn_v: torch.Tensor | None = None

        if self.is_cross_attn:
            attn_k, attn_v = cache.k, cache.v
            if attn_k.shape[1] != self.num_query_heads or attn_v.shape[1] != self.num_query_heads:
                raise ValueError(
                    f"Cross-attention cache head dimension ({attn_k.shape[1]}) "
                    f"does not match num_query_heads ({self.num_query_heads}). "
                    "Cache should be pre-repeated for GQA."
                )
        else:
            Xk_BxSxKxH = self.k_proj(Xkv)  # (B, S, K, H)
            Xv_BxSxKxH = self.v_proj(Xkv)  # (B, S, K, H)
            Xk_BxSxKxH = self.rotary_emb(Xk_BxSxKxH, position=kv_positions)  # (B, S, K, H)

            Xk_BxKxSxH = Xk_BxSxKxH.transpose(1, 2)  # (B, K, S, H)
            Xv_BxKxSxH = Xv_BxSxKxH.transpose(1, 2)  # (B, K, S, H)
            # S=1 for Decode Step

            Xk_BxNxSxH = Xk_BxKxSxH
            Xv_BxNxSxH = Xv_BxKxSxH

            if cache is None:
                attn_k = Xk_BxNxSxH
                attn_v = Xv_BxNxSxH
            else:
                if prefill:
                    attn_k, attn_v = Xk_BxNxSxH, Xv_BxNxSxH
                    cache.prefill(attn_k, attn_v)
                else:
                    attn_k, attn_v = cache.update_one(Xk_BxNxSxH, Xv_BxNxSxH)

        attn_output = F.scaled_dot_product_attention(
            Xq_BxNxTxH,
            attn_k,
            attn_v,
            attn_mask=attn_mask,
            scale=1.0,
            enable_gqa=self.num_gqa_groups > 1,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()  # (B, T, N, H)
        output = self.o_proj(attn_output)

        return output.to(original_dtype)


class EncoderLayer(nn.Module):
    """Transformer Encoder Layer using DenseGeneral."""

    def __init__(self, config: DiaConfig):
        super().__init__()
        self.config = config
        model_config = config.model
        enc_config = config.model.encoder
        embed_dim = enc_config.n_embd

        self.pre_sa_norm = RMSNorm(
            embed_dim,
            eps=model_config.normalization_layer_epsilon,
            dtype=torch.float32,
        )
        self.self_attention = Attention(
            config=config,
            q_embed_dim=embed_dim,
            kv_embed_dim=embed_dim,
            num_query_heads=enc_config.n_head,
            num_kv_heads=enc_config.n_head,
            head_dim=enc_config.head_dim,
            is_cross_attn=False,
            out_embed_dim=embed_dim,
        )
        self.post_sa_norm = RMSNorm(
            embed_dim,
            eps=model_config.normalization_layer_epsilon,
            dtype=torch.float32,
        )
        self.mlp = MlpBlock(
            config=config,
            embed_dim=embed_dim,
            intermediate_dim=enc_config.n_hidden,
        )

    def forward(
        self,
        x: torch.Tensor,
        src_positions: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = x
        x_norm = self.pre_sa_norm(x)

        sa_out = self.self_attention(
            Xq=x_norm,
            Xkv=x_norm,
            q_positions=src_positions,
            kv_positions=src_positions,
            attn_mask=attn_mask,
        )
        x = residual + sa_out

        residual = x
        x_norm = self.post_sa_norm(x)
        mlp_out = self.mlp(x_norm)
        x = residual + mlp_out

        return x


class Encoder(nn.Module):
    """Transformer Encoder Stack using DenseGeneral."""

    def __init__(self, config: DiaConfig):
        super().__init__()
        self.config = config
        model_config = config.model
        enc_config = config.model.encoder
        compute_dtype = _str_to_dtype(config.training.dtype)

        self.embedding = nn.Embedding(
            model_config.src_vocab_size,
            enc_config.n_embd,
            dtype=compute_dtype,
        )
        self.dropout = nn.Dropout(model_config.dropout)
        self.layers = nn.ModuleList([EncoderLayer(config=config) for _ in range(enc_config.n_layer)])
        self.norm = RMSNorm(
            enc_config.n_embd,
            eps=model_config.normalization_layer_epsilon,
            dtype=torch.float32,
        )

    def forward(
        self,
        x_ids: torch.Tensor,
        src_positions: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.embedding(x_ids)

        for layer in self.layers:
            x = layer(x, src_positions=src_positions, attn_mask=attn_mask)
        x = self.norm(x)
        return x


class DecoderLayer(nn.Module):
    """Transformer Decoder Layer using DenseGeneral."""

    def __init__(self, config: DiaConfig):
        super().__init__()
        self.config = config
        model_config = config.model
        dec_config = config.model.decoder
        enc_config = config.model.encoder
        dec_embed_dim = dec_config.n_embd
        enc_embed_dim = enc_config.n_embd

        # Norms
        self.pre_sa_norm = RMSNorm(
            dec_embed_dim,
            eps=model_config.normalization_layer_epsilon,
            dtype=torch.float32,
        )
        self.pre_ca_norm = RMSNorm(
            dec_embed_dim,
            eps=model_config.normalization_layer_epsilon,
            dtype=torch.float32,
        )
        self.pre_mlp_norm = RMSNorm(
            dec_embed_dim,
            eps=model_config.normalization_layer_epsilon,
            dtype=torch.float32,
        )

        # Self-Attention (GQA) with Causal Masking
        self.self_attention = Attention(
            config=config,
            q_embed_dim=dec_embed_dim,
            kv_embed_dim=dec_embed_dim,
            num_query_heads=dec_config.gqa_query_heads,
            num_kv_heads=dec_config.kv_heads,
            head_dim=dec_config.gqa_head_dim,
            is_cross_attn=False,
            out_embed_dim=dec_embed_dim,
        )
        # Cross-Attention (MHA)
        self.cross_attention = Attention(
            config=config,
            q_embed_dim=dec_embed_dim,
            kv_embed_dim=enc_embed_dim,  # Note kv_embed_dim
            num_query_heads=dec_config.cross_query_heads,
            num_kv_heads=dec_config.cross_query_heads,
            head_dim=dec_config.cross_head_dim,
            is_cross_attn=True,
            out_embed_dim=dec_embed_dim,
        )
        # MLP
        self.mlp = MlpBlock(
            config=config,
            embed_dim=dec_embed_dim,
            intermediate_dim=dec_config.n_hidden,
        )

    def forward(
        self,
        x: torch.Tensor,
        encoder_out: torch.Tensor,
        tgt_positions: torch.Tensor,
        src_positions: torch.Tensor | None,
        self_attn_mask: torch.Tensor,
        cross_attn_mask: torch.Tensor,
        self_attn_cache: KVCache,
        cross_attn_cache: KVCache,
        prefill: bool = False,
    ) -> torch.Tensor:
        residual = x
        x_norm = self.pre_sa_norm(x)

        sa_out = self.self_attention(
            Xq=x_norm,  # (2, 1, D)
            Xkv=x_norm,  # (2, 1, D)
            q_positions=tgt_positions,  # (2, 1)
            kv_positions=tgt_positions,  # (2, 1)
            attn_mask=self_attn_mask,  # (2, 1, 1, S_max)
            cache=self_attn_cache,
            prefill=prefill,
        )

        x = residual + sa_out

        residual = x
        x_norm = self.pre_ca_norm(x)
        ca_out = self.cross_attention(
            Xq=x_norm,
            Xkv=encoder_out,
            q_positions=tgt_positions,
            kv_positions=src_positions,
            attn_mask=cross_attn_mask,
            cache=cross_attn_cache,
        )
        x = residual + ca_out

        residual = x
        x_norm = self.pre_mlp_norm(x)
        mlp_out = self.mlp(x_norm)
        x = residual + mlp_out

        return x


class Decoder(nn.Module):
    """Transformer Decoder Stack using DenseGeneral."""

    def __init__(self, config: DiaConfig):
        super().__init__()
        self.config = config
        model_config = config.model
        dec_config = config.model.decoder
        data_config = config.data
        compute_dtype = _str_to_dtype(config.training.dtype)
        weight_dtype = _str_to_dtype(config.model.weight_dtype)
        self.num_channels = data_config.channels
        self.num_layers = dec_config.n_layer

        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(model_config.tgt_vocab_size, dec_config.n_embd, dtype=compute_dtype)
                for _ in range(self.num_channels)
            ]
        )
        self.dropout = nn.Dropout(model_config.dropout)
        self.layers = nn.ModuleList([DecoderLayer(config=config) for _ in range(self.num_layers)])
        self.norm = RMSNorm(
            dec_config.n_embd,
            eps=model_config.normalization_layer_epsilon,
            dtype=torch.float32,
        )

        self.logits_dense = DenseGeneral(
            in_shapes=(dec_config.n_embd,),
            out_features=(self.num_channels, model_config.tgt_vocab_size),
            axis=(-1,),
            weight_dtype=weight_dtype,
        )

    def precompute_cross_attention_kv(
        self,
        max_len: int,
        encoder_out: torch.Tensor,  # (B, S, E)
        src_positions: torch.Tensor | None,  # (B, S)
    ) -> list[KVCache]:
        """
        Computes the Key and Value tensors for cross-attention for each layer from the encoder output.
        """
        per_layer_kv_cache: list[KVCache] = []

        for layer in self.layers:
            cross_attn_module = layer.cross_attention
            k_proj = cross_attn_module.k_proj(encoder_out)
            v_proj = cross_attn_module.v_proj(encoder_out)

            k_proj = cross_attn_module.rotary_emb(k_proj, position=src_positions)
            k = k_proj.transpose(1, 2)
            v = v_proj.transpose(1, 2)

            per_layer_kv_cache.append(
                KVCache(
                    cross_attn_module.num_kv_heads,
                    max_len,
                    cross_attn_module.head_dim,
                    torch.bfloat16,
                    k.device,
                    k=k,
                    v=v,
                )
            )

        return per_layer_kv_cache

    def decode_step(
        self,
        tgt_ids_Bx1xC: torch.Tensor,  # [B, 1, C]
        tgt_pos_Bx1: torch.Tensor,  # [B, 1]
        encoder_out: torch.Tensor,  # [B, S, E]
        self_attn_mask: Any,  # None
        cross_attn_mask: torch.Tensor,  # [B, 1, 1, S]
        self_attention_cache: list[KVCache],
        cross_attention_cache: list[KVCache],
    ) -> torch.Tensor:
        """
        Performs a single decoding step, managing KV caches layer by layer.

        Returns:
            A tuple containing:
            - logits_Bx1xCV: The final output logits for the current step (B, 1, C*V), cast to float32.
        """
        assert self_attn_mask is None, "Self-attention mask should be None, kept for pattern"

        x = None
        for i in range(self.num_channels):
            channel_tokens = tgt_ids_Bx1xC[..., i]
            channel_embed = self.embeddings[i](channel_tokens)
            x = channel_embed if x is None else x + channel_embed

        for i, layer in enumerate(self.layers):
            self_cache = self_attention_cache[i]
            cross_cache = cross_attention_cache[i]
            x = layer(
                x,  # (2, 1, D)
                encoder_out,  # (2, S, E)
                src_positions=None,  # CA KV is already computed
                tgt_positions=tgt_pos_Bx1,  # (2, 1)
                self_attn_mask=None,
                cross_attn_mask=cross_attn_mask,
                self_attn_cache=self_cache,
                cross_attn_cache=cross_cache,
            )

        x = self.norm(x)
        logits_Bx1xCxV = self.logits_dense(x)

        return logits_Bx1xCxV.to(torch.float32)

    def forward(
        self,
        tgt_ids_BxTxC: torch.Tensor,
        encoder_out: torch.Tensor,
        tgt_positions: torch.Tensor,
        src_positions: torch.Tensor,
        self_attn_mask: torch.Tensor,
        cross_attn_mask: torch.Tensor,
        self_attention_cache: list[KVCache],
        cross_attention_cache: list[KVCache],
    ) -> torch.Tensor:
        """
        Forward pass for the Decoder stack, managing KV caches.

        Args:
            tgt_ids_BxTxC: Target token IDs (B, T, C).
            encoder_out: Output from the encoder (B, S, E).
            tgt_positions: Positions for target sequence (B, T).
            src_positions: Positions for source sequence (B, S).
            self_attn_mask: Mask for self-attention.
            cross_attn_mask: Mask for cross-attention.
            past_key_values: List containing the self-attention KV cache for each layer
                             from the previous decoding step. `len(past_key_values)` should
                             equal `num_layers`.
            precomputed_cross_attn_kv: A single tuple containing the pre-computed K/V cache
                                      derived from `encoder_out`. This is passed identically
                                      to all layers.

        Returns:
            A tuple containing:
            - logits: The final output logits (B, T, C * V), cast to float32.
            - present_key_values: A list containing the updated self-attention KV cache
                                 for each layer for the *current* decoding step.
        """
        _, _, num_channels_in = tgt_ids_BxTxC.shape
        assert num_channels_in == self.num_channels, "Input channels mismatch"

        # Embeddings
        x = None
        for i in range(self.num_channels):
            channel_tokens = tgt_ids_BxTxC[..., i]
            channel_embed = self.embeddings[i](channel_tokens)
            x = channel_embed if x is None else x + channel_embed

        for i, layer in enumerate(self.layers):
            x = layer(
                x,
                encoder_out,
                tgt_positions=tgt_positions,
                src_positions=src_positions,
                self_attn_mask=self_attn_mask,
                cross_attn_mask=cross_attn_mask,
                self_attn_cache=self_attention_cache[i],
                cross_attn_cache=cross_attention_cache[i],
                prefill=True,
            )

        # Final Norm
        x = self.norm(x)
        logits_BxTxCxV = self.logits_dense(x)

        return logits_BxTxCxV.to(torch.float32)


class DiaModel(nn.Module):
    """PyTorch Dia Model using DenseGeneral."""

    def __init__(self, config: DiaConfig):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(
        self,
        src_BxS: torch.Tensor,
        tgt_BxTxC: torch.Tensor,
        src_positions: torch.Tensor | None = None,
        tgt_positions: torch.Tensor | None = None,
        enc_self_attn_mask: torch.Tensor | None = None,
        dec_self_attn_mask: torch.Tensor | None = None,
        dec_cross_attn_mask: torch.Tensor | None = None,
    ):
        # --- Encoder Pass ---
        encoder_out = self.encoder(
            x_ids=src_BxS,
            src_positions=src_positions,
            attn_mask=enc_self_attn_mask,
        )

        # --- Decoder Pass ---
        logits = self.decoder(
            tgt_ids_BxTxC=tgt_BxTxC,
            encoder_out=encoder_out,
            tgt_positions=tgt_positions,
            src_positions=src_positions,
            self_attn_mask=dec_self_attn_mask,
            cross_attn_mask=dec_cross_attn_mask,
            precomputed_cross_attn_kv=None,
        )

        return logits
