import torch
import torch.nn as nn

class LSTM_Transformer_Hybrid(nn.Module):
    def __init__(self, num_features, num_heads, num_layers, lstm_hidden_size, lstm_layers, d_model=128, dropout=0.3):
        super(LSTM_Transformer_Hybrid, self).__init__()
        
        # LSTM layer to capture short-term dependencies
        self.lstm = nn.LSTM(input_size=num_features, hidden_size=lstm_hidden_size, num_layers=lstm_layers, batch_first=True)

        # Transformer encoder layer to capture long-term dependencies
        self.input_projection = nn.Linear(lstm_hidden_size, d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        # Final decoder to project to 1-dimensional output
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, src):
        # src shape: (batch_size, seq_length, num_features)

        # Pass input through LSTM layer
        lstm_out, _ = self.lstm(src)  # lstm_out shape: (batch_size, seq_length, lstm_hidden_size)
        
        # Project LSTM output to d_model size for the Transformer
        projected = self.input_projection(lstm_out)  # projected shape: (batch_size, seq_length, d_model)
        
        # Pass through Transformer encoder
        transformer_out = self.transformer_encoder(projected)  # transformer_out shape: (batch_size, seq_length, d_model)

        # Mean-pooling over the sequence dimension and decode to a single output
        output = self.decoder(transformer_out.mean(dim=1))  # output shape: (batch_size, 1)
        
        return output
