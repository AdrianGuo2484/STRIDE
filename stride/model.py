from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _triple(value: int | Sequence[int]) -> tuple[int, int, int]:
    if isinstance(value, int):
        return (value, value, value)
    result = tuple(int(item) for item in value)
    if len(result) != 3:
        raise ValueError(f"expected three spatial values, got {result}")
    return result


class InputAdapter(nn.Module):
    """Two Conv3D-BN-GELU layers followed by input-adaptation dropout."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.GELU(),
            nn.Dropout3d(dropout) if dropout > 0.0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MaskConditionedSoftGate(nn.Module):
    """Eqs. (2)-(3): learn a lesion-conditioned residual feature gate."""

    def __init__(self, feature_channels: int) -> None:
        super().__init__()
        self.prior_projection = nn.Conv3d(1, int(feature_channels), kernel_size=1)

    def forward(self, features: torch.Tensor, lesion_prior: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if lesion_prior.ndim == 4:
            lesion_prior = lesion_prior.unsqueeze(1)
        if lesion_prior.ndim != 5 or lesion_prior.shape[1] != 1:
            raise ValueError(f"lesion_prior must have shape [B,1,D,H,W], got {tuple(lesion_prior.shape)}")
        prior = lesion_prior.to(device=features.device, dtype=features.dtype).clamp(0.0, 1.0)
        if prior.shape[2:] != features.shape[2:]:
            prior = F.interpolate(prior, size=features.shape[2:], mode="trilinear", align_corners=False)
        gate = torch.sigmoid(features * self.prior_projection(prior))
        return features + (features * gate), gate


class DropPath(nn.Module):
    """Per-sample stochastic depth for residual Transformer branches."""

    def __init__(self, probability: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= probability < 1.0:
            raise ValueError(f"drop-path probability must be in [0, 1), got {probability}")
        self.probability = float(probability)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.probability == 0.0:
            return x
        keep_probability = 1.0 - self.probability
        shape = (x.shape[0],) + ((1,) * (x.ndim - 1))
        mask = x.new_empty(shape).bernoulli_(keep_probability)
        return x * mask / keep_probability


def _window_partition(
    x: torch.Tensor,
    window_size: tuple[int, int, int],
) -> tuple[torch.Tensor, tuple[int, int, int], tuple[int, int, int]]:
    # x is [B,D,H,W,C]. Padding is removed by _window_reverse.
    batch, depth, height, width, channels = x.shape
    wd, wh, ww = window_size
    padded = (
        int(math.ceil(depth / wd) * wd),
        int(math.ceil(height / wh) * wh),
        int(math.ceil(width / ww) * ww),
    )
    pd, ph, pw = padded[0] - depth, padded[1] - height, padded[2] - width
    x = x.permute(0, 4, 1, 2, 3)
    x = F.pad(x, (0, pw, 0, ph, 0, pd))
    x = x.permute(0, 2, 3, 4, 1).contiguous()
    dp, hp, wp = padded
    windows = (
        x.view(batch, dp // wd, wd, hp // wh, wh, wp // ww, ww, channels)
        .permute(0, 1, 3, 5, 2, 4, 6, 7)
        .reshape(-1, wd * wh * ww, channels)
    )
    return windows, (depth, height, width), padded


def _window_reverse(
    windows: torch.Tensor,
    window_size: tuple[int, int, int],
    original_shape: tuple[int, int, int],
    padded_shape: tuple[int, int, int],
    batch_size: int,
) -> torch.Tensor:
    wd, wh, ww = window_size
    dp, hp, wp = padded_shape
    channels = windows.shape[-1]
    x = (
        windows.view(batch_size, dp // wd, hp // wh, wp // ww, wd, wh, ww, channels)
        .permute(0, 1, 4, 2, 5, 3, 6, 7)
        .reshape(batch_size, dp, hp, wp, channels)
    )
    depth, height, width = original_shape
    return x[:, :depth, :height, :width].contiguous()


class WindowAttention3D(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int | Sequence[int],
        dropout: float,
        *,
        shifted: bool,
    ) -> None:
        super().__init__()
        self.window_size = _triple(window_size)
        self.num_heads = int(num_heads)
        self.shifted = bool(shifted)
        self.position = nn.Parameter(torch.zeros(1, *self.window_size, dim))
        nn.init.trunc_normal_(self.position, std=0.02)
        self.attention = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

    @staticmethod
    def _attention_mask(
        spatial_shape: tuple[int, int, int],
        window_size: tuple[int, int, int],
        shifts: tuple[int, int, int],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        padded = tuple(
            int(math.ceil(size / window) * window)
            for size, window in zip(spatial_shape, window_size)
        )
        mask = torch.zeros((1, *padded, 1), device=device, dtype=torch.int64)

        def regions(size: int, window: int, shift: int) -> tuple[slice, ...]:
            if shift == 0:
                return (slice(0, size),)
            return (slice(0, -window), slice(-window, -shift), slice(-shift, None))

        counter = 0
        for depth_slice in regions(padded[0], window_size[0], shifts[0]):
            for height_slice in regions(padded[1], window_size[1], shifts[1]):
                for width_slice in regions(padded[2], window_size[2], shifts[2]):
                    mask[:, depth_slice, height_slice, width_slice, :] = counter
                    counter += 1
        windows, _, _ = _window_partition(mask, window_size)
        windows = windows.squeeze(-1)
        differences = windows.unsqueeze(1) - windows.unsqueeze(2)
        return differences.ne(0).to(dtype=dtype) * torch.finfo(dtype).min

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        effective_window = tuple(min(size, spatial) for size, spatial in zip(self.window_size, x.shape[1:4]))
        shifts = tuple(
            window // 2 if self.shifted and spatial > window else 0
            for window, spatial in zip(effective_window, x.shape[1:4])
        )
        if any(shifts):
            x = torch.roll(x, shifts=tuple(-value for value in shifts), dims=(1, 2, 3))
        windows, original, padded = _window_partition(x, effective_window)
        position = self.position.permute(0, 4, 1, 2, 3)
        if effective_window != self.window_size:
            position = F.interpolate(position, size=effective_window, mode="trilinear", align_corners=False)
        position = position.permute(0, 2, 3, 4, 1).reshape(1, -1, windows.shape[-1])
        windows = windows + position
        attention_mask = None
        if any(shifts):
            attention_mask = self._attention_mask(
                original,
                effective_window,
                shifts,
                device=x.device,
                dtype=x.dtype,
            )
            attention_mask = attention_mask.repeat(x.shape[0], 1, 1)
            attention_mask = attention_mask.repeat_interleave(self.num_heads, dim=0)
        attended, _ = self.attention(
            windows,
            windows,
            windows,
            attn_mask=attention_mask,
            need_weights=False,
        )
        result = _window_reverse(attended, effective_window, original, padded, x.shape[0])
        if any(shifts):
            result = torch.roll(result, shifts=shifts, dims=(1, 2, 3))
        return result


class WindowTransformerBlock3D(nn.Module):
    """Regular or shifted fixed-window 3D Transformer block."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int | Sequence[int],
        *,
        shifted: bool,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"feature dim {dim} must be divisible by {num_heads} heads")
        self.norm_attention = nn.LayerNorm(dim)
        self.attention = WindowAttention3D(dim, num_heads, window_size, dropout, shifted=shifted)
        self.drop_path = DropPath(drop_path)
        self.norm_mlp = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attention(self.norm_attention(x)))
        return x + self.drop_path(self.mlp(self.norm_mlp(x)))


class MultiScaleAdaptiveSpatialWindow(nn.Module):
    """Eqs. (5)-(7): examination-specific continuous spatial weighting."""

    def __init__(self, in_channels: int, hidden_channels: int = 24, scales: Sequence[int] = (1, 2, 4)) -> None:
        super().__init__()
        if not scales or any(int(scale) < 1 for scale in scales):
            raise ValueError(f"localization scales must be positive, got {scales}")
        self.scales = tuple(int(scale) for scale in scales)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(
                        in_channels,
                        hidden_channels,
                        kernel_size=3,
                        stride=scale,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm3d(hidden_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(hidden_channels, 1, kernel_size=1),
                )
                for scale in self.scales
            ]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        responses = []
        for branch in self.branches:
            response = branch(x)
            if response.shape[2:] != x.shape[2:]:
                response = F.interpolate(response, size=x.shape[2:], mode="trilinear", align_corners=False)
            responses.append(response)
        spatial_window = torch.sigmoid(torch.stack(responses, dim=0).mean(dim=0))
        return x * spatial_window, spatial_window


class AWHTEncoder(nn.Module):
    """Adaptive spatial weighting followed by hierarchical local attention."""

    def __init__(
        self,
        in_channels: int,
        *,
        embed_dim: int = 48,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        patch_size: int | Sequence[int] = 4,
        attention_window_size: int | Sequence[int] = 8,
        localization_scales: Sequence[int] = (1, 2, 4),
        localization_hidden_channels: int = 24,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.15,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        if len(depths) != len(num_heads):
            raise ValueError("depths and num_heads must have the same length")
        self.patch_size = _triple(patch_size)
        self.use_checkpoint = bool(use_checkpoint)
        self.stage_dims = [int(embed_dim * (2**index)) for index in range(len(depths))]
        self.patch_embed = nn.Conv3d(
            in_channels,
            self.stage_dims[0],
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.adaptive_window = MultiScaleAdaptiveSpatialWindow(
            in_channels,
            localization_hidden_channels,
            localization_scales,
        )
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        total_blocks = sum(int(depth) for depth in depths)
        block_index = 0
        for index, (depth, heads) in enumerate(zip(depths, num_heads)):
            blocks = nn.ModuleList(
                [
                    WindowTransformerBlock3D(
                        self.stage_dims[index],
                        int(heads),
                        attention_window_size,
                        shifted=local_index % 2 == 1,
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        drop_path=(drop_path * (block_index + local_index) / max(total_blocks - 1, 1)),
                    )
                    for local_index in range(int(depth))
                ]
            )
            block_index += int(depth)
            self.stages.append(blocks)
            if index < len(depths) - 1:
                self.downsamples.append(
                    nn.Conv3d(self.stage_dims[index], self.stage_dims[index + 1], kernel_size=2, stride=2)
                )
        self.output_norm = nn.LayerNorm(self.stage_dims[-1])
        self.output_dim = self.stage_dims[-1]

    def forward(self, x: torch.Tensor) -> dict[str, object]:
        x, adaptive_window = self.adaptive_window(x)
        x = self.patch_embed(x).permute(0, 2, 3, 4, 1).contiguous()
        pyramid: list[torch.Tensor] = []
        for stage_index, blocks in enumerate(self.stages):
            for block in blocks:
                if self.use_checkpoint and self.training and x.requires_grad:
                    x = checkpoint(block, x, use_reentrant=False)
                else:
                    x = block(x)
            feature_map = x.permute(0, 4, 1, 2, 3).contiguous()
            pyramid.append(feature_map)
            if stage_index < len(self.downsamples):
                x = self.downsamples[stage_index](feature_map).permute(0, 2, 3, 4, 1).contiguous()
        pooled = self.output_norm(x.mean(dim=(1, 2, 3)))
        return {
            "latent": pooled,
            "feature_map": pyramid[-1],
            "pyramid": pyramid,
            "adaptive_window": adaptive_window,
        }


class TimeEmbedding(nn.Module):
    """Eq. (11): log-scale the interval and add periodic features."""

    def __init__(self, embedding_dim: int = 64, num_frequencies: int = 8, max_weeks: float = 52.0) -> None:
        super().__init__()
        if max_weeks <= 0:
            raise ValueError("max_weeks must be positive")
        self.max_weeks = float(max_weeks)
        frequencies = torch.arange(1, num_frequencies + 1, dtype=torch.float32)
        self.register_buffer("frequencies", frequencies, persistent=False)
        input_dim = 1 + (2 * num_frequencies)
        self.projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def normalize(self, delta_weeks: torch.Tensor) -> torch.Tensor:
        delta = delta_weeks.float().clamp_min(0.0).reshape(-1, 1)
        return torch.log1p(delta / self.max_weeks)

    def forward(self, delta_weeks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = self.normalize(delta_weeks)
        phase = math.pi * normalized * self.frequencies.reshape(1, -1)
        features = torch.cat([normalized, torch.sin(phase), torch.cos(phase)], dim=-1)
        return self.projection(features), normalized


class GatedLatentTransition(nn.Module):
    """Eqs. (15)-(16): interval-scaled, element-wise gated latent evolution."""

    def __init__(self, latent_dim: int, context_dim: int, time_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        input_dim = latent_dim + context_dim + time_dim
        self.dynamics = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.delta_projection = nn.Linear(hidden_dim, latent_dim)
        self.gate_projection = nn.Linear(hidden_dim, latent_dim)

    def forward(
        self,
        z_t: torch.Tensor,
        context: torch.Tensor,
        time_embedding: torch.Tensor,
        normalized_delta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.dynamics(torch.cat([z_t, context, time_embedding], dim=-1))
        latent_change = self.delta_projection(hidden)
        transition_gate = torch.sigmoid(self.gate_projection(hidden))
        z_pred = z_t + normalized_delta * transition_gate * latent_change
        return z_pred, latent_change, transition_gate


class LongitudinalFusion(nn.Module):
    """Eqs. (18)-(19): project, fuse, and aggregate seven longitudinal sources."""

    def __init__(
        self,
        source_dims: Sequence[int],
        latent_dim: int,
        refinement_dim: int,
        num_heads: int,
        depth: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if latent_dim % num_heads != 0:
            raise ValueError(f"fusion latent dim {latent_dim} must be divisible by {num_heads} heads")
        self.projections = nn.ModuleList([nn.Linear(int(source_dim), latent_dim) for source_dim in source_dims])
        layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=latent_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.operator = nn.TransformerEncoder(
            layer,
            num_layers=int(depth),
            norm=nn.LayerNorm(latent_dim),
            enable_nested_tensor=False,
        )
        self.refinement = nn.Sequential(
            nn.Linear(latent_dim, refinement_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(refinement_dim),
        )
        self.output_dim = int(refinement_dim)

    def forward(self, sources: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if len(sources) != len(self.projections):
            raise ValueError(f"expected {len(self.projections)} fusion sources, got {len(sources)}")
        source_tokens = torch.stack(
            [projection(source) for projection, source in zip(self.projections, sources)],
            dim=1,
        )
        fused_tokens = self.operator(source_tokens)
        return self.refinement(fused_tokens.mean(dim=1)), fused_tokens


class STRIDE(nn.Module):
    """Paper-aligned longitudinal classifier implementing Eqs. (1)-(23)."""

    def __init__(
        self,
        *,
        num_modalities: int = 4,
        stem_channels: int = 12,
        input_dropout: float = 0.08,
        embed_dim: int = 48,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        patch_size: int | Sequence[int] = 4,
        attention_window_size: int | Sequence[int] = 8,
        localization_scales: Sequence[int] = (1, 2, 4),
        localization_hidden_channels: int = 24,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        drop_path: float = 0.15,
        use_checkpoint: bool = False,
        time_embedding_dim: int = 64,
        time_frequencies: int = 8,
        max_weeks: float = 52.0,
        context_dim: int = 192,
        dynamics_hidden_dim: int = 384,
        fusion_heads: int = 8,
        fusion_depth: int = 1,
        refinement_dim: int = 384,
        num_lumiere_classes: int = 7,
        num_burdenko_classes: int = 3,
    ) -> None:
        super().__init__()
        self.num_modalities = int(num_modalities)
        packed_channels = self.num_modalities * 2
        self.visit_projection = nn.Conv3d(packed_channels, stem_channels, kernel_size=1)
        self.soft_gate = MaskConditionedSoftGate(stem_channels)
        self.input_adapter = InputAdapter(stem_channels, stem_channels, input_dropout)
        self.encoder = AWHTEncoder(
            stem_channels,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            patch_size=patch_size,
            attention_window_size=attention_window_size,
            localization_scales=localization_scales,
            localization_hidden_channels=localization_hidden_channels,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            drop_path=drop_path,
            use_checkpoint=use_checkpoint,
        )
        latent_dim = self.encoder.output_dim
        self.time_embedding = TimeEmbedding(time_embedding_dim, time_frequencies, max_weeks)
        self.interaction_gate = nn.Linear(latent_dim * 3, latent_dim)
        context_input_dim = latent_dim * 4
        self.pair_context = nn.Sequential(
            nn.LayerNorm(context_input_dim),
            nn.Linear(context_input_dim, dynamics_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dynamics_hidden_dim, context_dim),
        )
        self.transition = GatedLatentTransition(
            latent_dim,
            context_dim,
            time_embedding_dim,
            dynamics_hidden_dim,
            dropout,
        )
        self.fusion = LongitudinalFusion(
            [latent_dim, latent_dim, latent_dim, latent_dim, latent_dim, context_dim, time_embedding_dim],
            latent_dim,
            refinement_dim,
            fusion_heads,
            fusion_depth,
            dropout,
        )
        self.lumiere_head = nn.Linear(self.fusion.output_dim, num_lumiere_classes)
        self.burdenko_head = nn.Linear(self.fusion.output_dim, num_burdenko_classes)
        self._transferred_modules_trainable = True

    def _pack_visit(
        self,
        x: torch.Tensor,
        modality_presence: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 5 or x.shape[1] != self.num_modalities:
            raise ValueError(
                f"MRI input must have shape [B,{self.num_modalities},D,H,W], got {tuple(x.shape)}"
            )
        if modality_presence.shape != (x.shape[0], self.num_modalities):
            raise ValueError(
                f"modality_presence must have shape {(x.shape[0], self.num_modalities)}, "
                f"got {tuple(modality_presence.shape)}"
            )
        auxiliary = modality_presence.to(dtype=x.dtype, device=x.device)[:, :, None, None, None].expand_as(x)
        return torch.cat([x, auxiliary], dim=1)

    def _encode_visit(
        self,
        x: torch.Tensor,
        lesion_prior: torch.Tensor,
        modality_presence: torch.Tensor,
    ) -> dict[str, object]:
        packed = self._pack_visit(x, modality_presence)
        features = self.visit_projection(packed)
        gated, gate = self.soft_gate(features, lesion_prior)
        adapted = self.input_adapter(gated)
        encoded = self.encoder(adapted)
        encoded["soft_gate"] = gate
        return encoded

    def forward(
        self,
        x_t: torch.Tensor,
        x_next: torch.Tensor,
        lesion_t: torch.Tensor,
        lesion_next: torch.Tensor,
        modality_t: torch.Tensor,
        modality_next: torch.Tensor,
        delta_weeks: torch.Tensor,
        *,
        task: str,
    ) -> dict[str, object]:
        current = self._encode_visit(x_t, lesion_t, modality_t)
        follow_up = self._encode_visit(x_next, lesion_next, modality_next)
        z_t = current["latent"]
        z_next = follow_up["latent"]
        assert isinstance(z_t, torch.Tensor) and isinstance(z_next, torch.Tensor)
        time_embedding, normalized_delta = self.time_embedding(delta_weeks.to(x_t.device))
        observed_change = z_next - z_t
        interaction_alpha = torch.sigmoid(
            self.interaction_gate(torch.cat([z_t, z_next, observed_change], dim=-1))
        )
        interaction = interaction_alpha * (z_t * z_next)
        context_input = torch.cat(
            [z_t, z_next, observed_change, interaction],
            dim=-1,
        )
        context = self.pair_context(context_input)
        z_pred, latent_change, transition_gate = self.transition(
            z_t,
            context,
            time_embedding,
            normalized_delta,
        )
        pair_representation, fusion_tokens = self.fusion(
            [
                z_t,
                z_next,
                z_pred,
                observed_change,
                z_pred - z_t,
                context,
                time_embedding,
            ]
        )
        if task == "lumiere":
            logits = self.lumiere_head(pair_representation)
        elif task == "burdenko":
            logits = self.burdenko_head(pair_representation)
        else:
            raise ValueError(f"task must be 'lumiere' or 'burdenko', got {task!r}")
        return {
            "logits": logits,
            "z_t": z_t,
            "z_next": z_next,
            "z_pred": z_pred,
            "latent_change": latent_change,
            "transition_gate": transition_gate,
            "interaction_gate": interaction_alpha,
            "pair_context": context,
            "fusion_tokens": fusion_tokens,
            "time_embedding": time_embedding,
            "normalized_delta": normalized_delta,
            "soft_gate_t": current["soft_gate"],
            "soft_gate_next": follow_up["soft_gate"],
            "adaptive_window_t": current["adaptive_window"],
            "adaptive_window_next": follow_up["adaptive_window"],
        }

    def set_transferred_modules_trainable(self, trainable: bool) -> None:
        """Freeze or unfreeze every module transferred from LUMIERE."""
        self._transferred_modules_trainable = bool(trainable)
        for module in self.transferred_modules():
            for parameter in module.parameters():
                parameter.requires_grad = trainable

    def transferred_modules(self) -> tuple[nn.Module, ...]:
        return (
            self.visit_projection,
            self.input_adapter,
            self.soft_gate,
            self.encoder,
            self.time_embedding,
            self.interaction_gate,
            self.pair_context,
            self.transition,
            self.fusion,
        )

    def enforce_frozen_module_eval(self) -> None:
        """Keep dropout and batch-normalization fixed while transferred modules are frozen."""
        if not self._transferred_modules_trainable:
            for module in self.transferred_modules():
                module.eval()


class AWHTSegmenter(nn.Module):
    """Parameter-independent BraTS PSN: AWHT encoder plus convolutional decoder."""

    def __init__(
        self,
        *,
        num_modalities: int = 4,
        num_regions: int = 3,
        stem_channels: int = 12,
        input_dropout: float = 0.08,
        embed_dim: int = 48,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        patch_size: int | Sequence[int] = 4,
        attention_window_size: int | Sequence[int] = 8,
        localization_scales: Sequence[int] = (1, 2, 4),
        localization_hidden_channels: int = 24,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        drop_path: float = 0.15,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.adapter = InputAdapter(num_modalities, stem_channels, input_dropout)
        self.encoder = AWHTEncoder(
            stem_channels,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            patch_size=patch_size,
            attention_window_size=attention_window_size,
            localization_scales=localization_scales,
            localization_hidden_channels=localization_hidden_channels,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            drop_path=drop_path,
            use_checkpoint=use_checkpoint,
        )
        dimensions = self.encoder.stage_dims
        decoder: list[nn.Module] = []
        current_dim = dimensions[-1]
        for target_dim in reversed(dimensions[:-1]):
            decoder.extend(
                [
                    nn.ConvTranspose3d(current_dim, target_dim, kernel_size=2, stride=2, bias=False),
                    nn.BatchNorm3d(target_dim),
                    nn.GELU(),
                ]
            )
            current_dim = target_dim
        decoder.extend(
            [
                nn.ConvTranspose3d(
                    current_dim,
                    stem_channels,
                    kernel_size=self.encoder.patch_size,
                    stride=self.encoder.patch_size,
                    bias=False,
                ),
                nn.BatchNorm3d(stem_channels),
                nn.GELU(),
            ]
        )
        self.decoder = nn.Sequential(*decoder)
        self.output = nn.Conv3d(stem_channels, int(num_regions), kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict[str, object]:
        target_shape = x.shape[2:]
        encoded = self.encoder(self.adapter(x))
        feature_map = encoded["feature_map"]
        assert isinstance(feature_map, torch.Tensor)
        logits = self.output(self.decoder(feature_map))
        if logits.shape[2:] != target_shape:
            logits = F.interpolate(logits, size=target_shape, mode="trilinear", align_corners=False)
        return {"logits": logits, "adaptive_window": encoded["adaptive_window"]}


def load_longitudinal_pretraining(model: STRIDE, checkpoint: dict[str, object]) -> tuple[list[str], list[str]]:
    """Transfer all longitudinal modules except task-specific classification heads."""
    state = checkpoint.get("model_state", checkpoint)
    if not isinstance(state, dict):
        raise TypeError("checkpoint must contain a model_state mapping")
    transferable = {
        key: value
        for key, value in state.items()
        if not key.startswith("lumiere_head.") and not key.startswith("burdenko_head.")
    }
    incompatible = model.load_state_dict(transferable, strict=False)
    return list(incompatible.missing_keys), list(incompatible.unexpected_keys)
