"""Temporal Fusion Transformer model implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network (GRN) block."""
    
    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(hidden_size, hidden_size)
        
        # Skip connection
        if input_size != hidden_size:
            self.skip_proj = nn.Linear(input_size, hidden_size)
        else:
            self.skip_proj = None
    
    def forward(self, x):
        # Feedforward
        hidden = self.linear1(x)
        hidden = F.elu(hidden)
        hidden = self.dropout(hidden)
        hidden = self.linear2(hidden)
        
        # Gating
        gate = torch.sigmoid(self.gate(hidden))
        hidden = hidden * gate
        
        # Skip connection
        if self.skip_proj is not None:
            x = self.skip_proj(x)
        
        return hidden + x


class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network for feature selection."""
    
    def __init__(self, input_size: int, hidden_size: int, num_vars: int, dropout: float = 0.1):
        super().__init__()
        self.num_vars = num_vars
        self.hidden_size = hidden_size
        
        # Feature processing
        self.feature_grns = nn.ModuleList([
            GatedResidualNetwork(input_size, hidden_size, dropout)
            for _ in range(num_vars)
        ])
        
        # Variable selection weights
        self.selection_grn = GatedResidualNetwork(num_vars * hidden_size, hidden_size, dropout)
        self.selection_weights = nn.Linear(hidden_size, num_vars)
        
    def forward(self, x):
        # x: [batch, seq_len, num_vars * feature_size] or [batch, num_vars * feature_size]
        batch_size = x.size(0)
        if x.dim() == 3:
            seq_len = x.size(1)
            x = x.view(batch_size * seq_len, -1)
            reshape_output = True
        else:
            reshape_output = False
        
        # Process each variable
        var_features = []
        feature_size = x.size(-1) // self.num_vars
        
        for i in range(self.num_vars):
            start_idx = i * feature_size
            end_idx = (i + 1) * feature_size
            var_input = x[:, start_idx:end_idx]
            var_feat = self.feature_grns[i](var_input)
            var_features.append(var_feat)
        
        # Concatenate and compute selection weights
        all_features = torch.cat(var_features, dim=-1)
        selection_input = self.selection_grn(all_features)
        weights = F.softmax(self.selection_weights(selection_input), dim=-1)
        
        # Apply weights
        weighted_features = []
        for i, feat in enumerate(var_features):
            weight = weights[:, i:i+1]
            weighted_features.append(feat * weight)
        
        output = torch.stack(weighted_features, dim=-1).sum(dim=-1)
        
        if reshape_output:
            output = output.view(batch_size, seq_len, -1)
        
        return output


class TFTForecast(nn.Module):
    """Simplified Temporal Fusion Transformer for forecasting."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_heads: int = 4,
        encoder_layers: int = 1,
        decoder_layers: int = 1,
        horizon: int = 4,
        dropout: float = 0.1
    ):
        """Initialize TFT model.
        
        Args:
            input_size: Number of input features
            hidden_size: Hidden layer size
            num_heads: Number of attention heads
            encoder_layers: Number of encoder layers
            decoder_layers: Number of decoder layers
            horizon: Forecast horizon
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.horizon = horizon
        
        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # Variable selection (simplified)
        self.variable_selection = VariableSelectionNetwork(
            input_size, hidden_size, min(input_size, 10), dropout
        )
        
        # Encoder LSTM
        self.encoder = nn.LSTM(
            hidden_size, hidden_size, encoder_layers,
            batch_first=True, dropout=dropout if encoder_layers > 1 else 0
        )
        
        # Decoder LSTM
        self.decoder = nn.LSTM(
            hidden_size, hidden_size, decoder_layers,
            batch_first=True, dropout=dropout if decoder_layers > 1 else 0
        )
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon)
        )
        
        # Position encoding
        self.pos_encoding = PositionalEncoding(hidden_size, dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor [batch_size, sequence_length, input_size]
            
        Returns:
            Output tensor [batch_size, horizon]
        """
        batch_size, seq_len, _ = x.shape
        
        # Variable selection
        x_selected = self.variable_selection(x)
        
        # Add position encoding
        x_pos = self.pos_encoding(x_selected)
        
        # Encoder
        encoder_out, (h_n, c_n) = self.encoder(x_pos)
        
        # Decoder (using last hidden state as input)
        decoder_input = h_n[-1].unsqueeze(1)  # [batch, 1, hidden]
        decoder_out, _ = self.decoder(decoder_input, (h_n, c_n))
        
        # Self-attention on encoder outputs
        attn_out, _ = self.attention(decoder_out, encoder_out, encoder_out)
        
        # Output projection
        output = self.output_proj(attn_out.squeeze(1))  # [batch, horizon]
        
        return output


class PositionalEncoding(nn.Module):
    """Positional encoding for sequences."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)