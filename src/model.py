from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch import nn


def _build_output_activation(name: Optional[str]) -> Optional[nn.Module]:
    key = str(name or "").lower()
    if key in {"", "none"}:
        return None
    if key == "softplus":
        return nn.Softplus()
    if key == "relu":
        return nn.ReLU()
    if key == "sigmoid":
        return nn.Sigmoid()
    raise ValueError("Unsupported output activation: {0}".format(name))


class GRUBackbone(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        use_attention: bool = False,
        attention_hidden_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.use_attention = bool(use_attention)
        if self.use_attention:
            attn_hidden = attention_hidden_size or hidden_size
            self.attention = nn.Sequential(
                nn.Linear(hidden_size, attn_hidden),
                nn.Tanh(),
                nn.Linear(attn_hidden, 1),
            )
            self.output_size = hidden_size * 2
        else:
            self.attention = None
            self.output_size = hidden_size
        self._reset_parameters()
        self._last_attention_weights: Optional[torch.Tensor] = None

    def _reset_parameters(self) -> None:
        for name, param in self.gru.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        if self.attention is not None:
            for module in self.attention:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> torch.Tensor:
        gru_out, hidden_state = self.gru(x, hidden)
        last_hidden = gru_out[:, -1, :]
        if self.use_attention and self.attention is not None:
            attn_scores = self.attention(gru_out).squeeze(-1)
            attn_weights = torch.softmax(attn_scores, dim=1)
            context = torch.sum(gru_out * attn_weights.unsqueeze(-1), dim=1)
            combined = torch.cat([last_hidden, context], dim=1)
            self._last_attention_weights = attn_weights.detach().cpu()
        else:
            combined = last_hidden
            self._last_attention_weights = None
        combined = self.dropout(combined)
        return combined

    @property
    def last_attention_weights(self) -> Optional[torch.Tensor]:
        return self._last_attention_weights


class GRUForecaster(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        horizon: int,
        output_activation: Optional[str] = None,
        use_attention: bool = False,
        attention_hidden_size: Optional[int] = None,
        use_horizon_embedding: bool = False,
        horizon_embed_dim: int = 8,
    ) -> None:
        super().__init__()
        self.backbone = GRUBackbone(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            use_attention=use_attention,
            attention_hidden_size=attention_hidden_size,
        )
        self.horizon = horizon
        self.use_horizon_embedding = use_horizon_embedding
        self.horizon_embed_dim = horizon_embed_dim
        
        if self.use_horizon_embedding:
            # Horizon-specific decoders with embeddings
            self.horizon_embed = nn.Embedding(horizon, horizon_embed_dim)
            self.horizon_decoders = nn.ModuleList([
                nn.Linear(self.backbone.output_size + horizon_embed_dim, 1)
                for _ in range(horizon)
            ])
            # Initialize decoders
            for decoder in self.horizon_decoders:
                nn.init.xavier_uniform_(decoder.weight)
                nn.init.zeros_(decoder.bias)
        else:
            # Standard single decoder
            self.fc = nn.Linear(self.backbone.output_size, horizon)
            nn.init.xavier_uniform_(self.fc.weight)
            nn.init.zeros_(self.fc.bias)
        
        self.output_activation = _build_output_activation(output_activation)

    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> torch.Tensor:
        features = self.backbone(x, hidden)
        
        if self.use_horizon_embedding:
            # Horizon-specific decoding
            outputs = []
            for h in range(self.horizon):
                # Get horizon embedding
                h_idx = torch.tensor([h], device=x.device, dtype=torch.long)
                h_embed = self.horizon_embed(h_idx)  # [1, embed_dim]
                h_embed = h_embed.expand(x.size(0), -1)  # [batch, embed_dim]
                
                # Concatenate features with horizon embedding
                combined = torch.cat([features, h_embed], dim=1)  # [batch, features + embed]
                
                # Predict for this horizon
                pred_h = self.horizon_decoders[h](combined)  # [batch, 1]
                outputs.append(pred_h)
            
            output = torch.cat(outputs, dim=1)  # [batch, horizon]
        else:
            # Standard decoding
            output = self.fc(features)
        
        if self.output_activation is not None:
            output = self.output_activation(output)
        return output

    @property
    def last_attention_weights(self) -> Optional[torch.Tensor]:
        return self.backbone.last_attention_weights


class GRUMultiHeadForecaster(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        horizon: int,
        output_activation: Optional[str] = None,
        use_attention: bool = False,
        attention_hidden_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.backbone = GRUBackbone(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            use_attention=use_attention,
            attention_hidden_size=attention_hidden_size,
        )
        self.heads = nn.ModuleList([
            nn.Linear(self.backbone.output_size, 1) for _ in range(horizon)
        ])
        for head in self.heads:
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)
        self.horizon = horizon
        self.output_activation = _build_output_activation(output_activation)

    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> torch.Tensor:
        features = self.backbone(x, hidden)
        outputs = [head(features) for head in self.heads]
        output = torch.cat(outputs, dim=1)
        if self.output_activation is not None:
            output = self.output_activation(output)
        return output

    @property
    def last_attention_weights(self) -> Optional[torch.Tensor]:
        return self.backbone.last_attention_weights


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size]


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self) -> None:
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs: int, num_channels: List[int], kernel_size: int, dropout: float) -> None:
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation
            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation,
                    padding=padding,
                    dropout=dropout,
                )
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TCNForecaster(nn.Module):
    def __init__(
        self,
        input_size: int,
        horizon: int,
        channels: List[int],
        kernel_size: int,
        dropout: float,
        output_activation: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.tcn = TemporalConvNet(input_size, channels, kernel_size, dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(channels[-1], horizon)
        self.output_activation = _build_output_activation(output_activation)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, features)
        tcn_input = x.transpose(1, 2)
        tcn_out = self.tcn(tcn_input)
        features = self.dropout(tcn_out[:, :, -1])
        output = self.fc(features)
        if self.output_activation is not None:
            output = self.output_activation(output)
        return output


class Seq2SeqForecaster(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        horizon: int,
        output_activation: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.encoder = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.decoder = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_layer = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
        self.horizon = horizon
        self.output_activation = _build_output_activation(output_activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        encoder_outputs, hidden = self.encoder(x)
        decoder_input = torch.zeros(batch_size, 1, hidden.size(-1), device=x.device)
        outputs = []
        for _ in range(self.horizon):
            decoder_output, hidden = self.decoder(decoder_input, hidden)
            decoder_output = self.dropout(decoder_output)
            step = self.output_layer(decoder_output.squeeze(1))
            outputs.append(step)
            decoder_input = decoder_output
        output = torch.cat(outputs, dim=1)
        if self.output_activation is not None:
            output = self.output_activation(output)
        return output


def create_model_from_config(input_size: int, horizon: int, model_cfg: Dict) -> nn.Module:
    architecture = str(model_cfg.get("architecture", "gru")).lower()
    hidden_size = int(model_cfg["hidden_size"])
    num_layers = int(model_cfg.get("num_layers", 1))
    dropout = float(model_cfg.get("dropout", 0.0))
    output_activation = model_cfg.get("output_activation")
    
    # Horizon embedding settings
    use_horizon_embedding = bool(model_cfg.get("use_horizon_embedding", False))
    horizon_embed_dim = int(model_cfg.get("horizon_embed_dim", 8))

    if architecture in {"gru", "gru_attention"}:
        return GRUForecaster(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            horizon=horizon,
            output_activation=output_activation,
            use_attention=model_cfg.get("use_attention", architecture == "gru_attention"),
            attention_hidden_size=model_cfg.get("attention_hidden_size"),
            use_horizon_embedding=use_horizon_embedding,
            horizon_embed_dim=horizon_embed_dim,
        )
    if architecture == "gru_multihead":
        return GRUMultiHeadForecaster(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            horizon=horizon,
            output_activation=output_activation,
            use_attention=model_cfg.get("use_attention", False),
            attention_hidden_size=model_cfg.get("attention_hidden_size"),
        )
    if architecture == "tcn":
        channels = model_cfg.get("tcn_channels", [hidden_size, hidden_size])
        kernel_size = int(model_cfg.get("tcn_kernel_size", 3))
        return TCNForecaster(
            input_size=input_size,
            horizon=horizon,
            channels=[int(ch) for ch in channels],
            kernel_size=kernel_size,
            dropout=dropout,
            output_activation=output_activation,
        )
    if architecture == "seq2seq":
        return Seq2SeqForecaster(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            horizon=horizon,
            output_activation=output_activation,
        )
    raise ValueError("Unsupported architecture: {0}".format(architecture))


__all__ = [
    "GRUForecaster",
    "GRUMultiHeadForecaster",
    "TCNForecaster",
    "Seq2SeqForecaster",
    "create_model_from_config",
]
