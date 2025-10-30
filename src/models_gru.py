"""GRU model implementation."""

import torch
import torch.nn as nn


class GRUForecast(nn.Module):
    """GRU model for multi-step forecasting with optional horizon embeddings."""
    
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int = 64, 
        num_layers: int = 2, 
        dropout: float = 0.2, 
        horizon: int = 4,
        output_activation: str = 'softplus',  # 'softplus' or None
        min_output: float = 0.0,              # minimum output for clipping
        max_output: float = None,             # maximum output for clipping
        use_horizon_embedding: bool = True,   # NEW: use horizon-specific decoders
        horizon_embed_dim: int = 8            # NEW: embedding dimension for horizons
    ):
        """Initialize GRU model.
        
        Args:
            input_size: Number of input features
            hidden_size: Hidden layer size
            num_layers: Number of GRU layers
            dropout: Dropout rate
            horizon: Forecast horizon
            output_activation: 'softplus' for non-negative outputs, None for unrestricted
            min_output: Minimum output value for post-processing clipping
            max_output: Maximum output value for post-processing clipping (None = no max)
            use_horizon_embedding: If True, use horizon-specific decoders with embeddings
            horizon_embed_dim: Dimension of horizon embeddings
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.horizon = horizon
        self.output_activation = output_activation
        self.min_output = min_output
        self.max_output = max_output
        self.use_horizon_embedding = use_horizon_embedding
        self.horizon_embed_dim = horizon_embed_dim
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output projection
        if self.use_horizon_embedding:
            # Horizon-specific decoders with embeddings
            self.horizon_embed = nn.Embedding(horizon, horizon_embed_dim)
            self.horizon_decoders = nn.ModuleList([
                nn.Linear(hidden_size + horizon_embed_dim, 1)
                for _ in range(horizon)
            ])
        else:
            # Standard single decoder for all horizons
            self.linear_out = nn.Linear(hidden_size, horizon)
        
        # Optional Softplus activation (only if specified)
        if self.output_activation == 'softplus':
            self.softplus = nn.Softplus()
        
    def forward(self, x: torch.Tensor, return_raw_output: bool = False) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor [batch_size, sequence_length, input_size]
            return_raw_output: If True, return raw output before activation/clipping
            
        Returns:
            Output tensor [batch_size, horizon] (processed or raw)
        """
        # GRU forward pass
        gru_out, hidden = self.gru(x)  # gru_out: [batch, seq, hidden]
        
        # Use last hidden state
        last_hidden = gru_out[:, -1, :]  # [batch, hidden]
        
        # Apply dropout
        last_hidden = self.dropout(last_hidden)
        
        # Generate predictions
        if self.use_horizon_embedding:
            # Horizon-specific decoding
            outputs = []
            for h in range(self.horizon):
                # Get horizon embedding
                h_idx = torch.tensor([h], device=x.device, dtype=torch.long)
                h_embed = self.horizon_embed(h_idx)  # [1, embed_dim]
                h_embed = h_embed.expand(x.size(0), -1)  # [batch, embed_dim]
                
                # Concatenate hidden state with horizon embedding
                combined = torch.cat([last_hidden, h_embed], dim=1)  # [batch, hidden + embed]
                
                # Predict for this horizon
                pred_h = self.horizon_decoders[h](combined)  # [batch, 1]
                outputs.append(pred_h)
            
            raw_output = torch.cat(outputs, dim=1)  # [batch, horizon]
        else:
            # Standard decoding
            raw_output = self.linear_out(last_hidden)  # [batch, horizon]
        
        if return_raw_output:
            return raw_output
        
        # Apply activation and/or clipping based on configuration
        output = raw_output
        
        # Apply activation if specified
        if self.output_activation == 'softplus':
            output = self.softplus(output)
        
        # Apply post-processing clipping
        if self.min_output is not None:
            output = torch.clamp(output, min=self.min_output)
        if self.max_output is not None:
            output = torch.clamp(output, max=self.max_output)
            
        return output