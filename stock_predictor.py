import yfinance as yf
import pandas as pd
import numpy as np
import pandas_datareader.data as web
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import ta
from textblob import TextBlob
import nltk

nltk.download('punkt')

# 1. Data Collection
def fetch_stock_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    treasury_yield = web.DataReader('DGS10', 'fred', start_date, end_date)
    treasury_yield = treasury_yield.resample('D').ffill()
    data = data.merge(treasury_yield, left_index=True, right_index=True, how='left')
    data.rename(columns={'DGS10': '10Y_Treasury_Yield'}, inplace=True)
    return data

# 2. Feature Engineering
def add_technical_indicators(data):
    # RSI, MACD, Bollinger Bands
    data['RSI'] = ta.momentum.RSIIndicator(close=data['Close']).rsi()
    macd = ta.trend.MACD(close=data['Close'])
    data['MACD'] = macd.macd_diff()
    bb = ta.volatility.BollingerBands(close=data['Close'])
    data['Bollinger_Width'] = bb.bollinger_wband()
    return data

def add_sentiment_analysis(data):
    # Simulate sentiment analysis (you would use real headlines in a production app)
    data['Headlines'] = "Apple stock performance"
    data['Sentiment'] = data['Headlines'].apply(lambda x: TextBlob(x).sentiment.polarity)
    return data

def add_lag_features(data):
    # Lag features for previous days' prices
    for lag in range(1, 6):
        data[f'Lag_{lag}'] = data['Close'].shift(lag)
    return data

def add_rolling_statistics(data):
    data['Rolling_Mean_20'] = data['Close'].rolling(window=20).mean()
    data['Rolling_Std_20'] = data['Close'].rolling(window=20).std()
    return data

# 3. Preprocess Data
def preprocess_data(data):
    data = add_technical_indicators(data)
    data = add_sentiment_analysis(data)
    data = add_lag_features(data)
    data = add_rolling_statistics(data)
    data.dropna(inplace=True)
    return data

# 4. Create Sequences for Time Series
def create_sequences(X, y, seq_length=60):
    sequences = []
    targets = []
    for i in range(len(X) - seq_length):
        seq = X[i:i + seq_length]
        target = y[i + seq_length]
        sequences.append(seq)
        targets.append(target)
    return torch.stack(sequences), torch.stack(targets)

# 5. Transformer Model with Linear Projection
class TimeSeriesTransformer(nn.Module):
    def __init__(self, num_features, num_heads, num_layers, d_model=64, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        # Project input features to d_model (must be divisible by num_heads)
        self.input_projection = nn.Linear(num_features, d_model)
        self.pos_encoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(d_model, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = self.input_projection(src)  # Project input to d_model size
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = output.mean(dim=1)
        output = self.decoder(output)
        return output

# 6. Training the Model
def train_model(model, train_loader, num_epochs=50, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}')

# 7. Evaluate Model
def evaluate_model(model, test_loader, target_scaler):
    model.eval()
    predictions, actuals = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            output = model(X_batch)
            predictions.append(output.detach().numpy())
            actuals.append(y_batch.numpy())

    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)
    predictions = target_scaler.inverse_transform(predictions)
    actuals = target_scaler.inverse_transform(actuals)

    return actuals, predictions

# 8. Plot Predictions
def plot_predictions(actuals, predictions):
    plt.figure(figsize=(12, 6))
    plt.plot(actuals, label='Actual Prices')
    plt.plot(predictions, label='Predicted Prices')
    plt.legend()
    plt.title('Actual vs Predicted Stock Prices')
    plt.xlabel('Time')
    plt.ylabel('Price')
    
    # Save the plot as a PNG image
    plt.savefig('predicted_vs_actual.png')
    
    # Optionally, close the plot to release memory
    plt.close()

    print("Plot saved as 'predicted_vs_actual.png'")


# 9. Main Workflow
if __name__ == "__main__":
    # Define parameters
    stock_symbol = 'AAPL'
    start_date = '2010-01-01'
    end_date = '2023-10-01'
    seq_length = 60
    batch_size = 32
    num_epochs = 10

    # Fetch and preprocess data
    data = fetch_stock_data(stock_symbol, start_date, end_date)
    data = preprocess_data(data)

    # Features and target
    features = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        '10Y_Treasury_Yield', 'RSI', 'MACD', 'Bollinger_Width',
        'Sentiment', 'Rolling_Mean_20', 'Rolling_Std_20',
        'Lag_1', 'Lag_2', 'Lag_3', 'Lag_4', 'Lag_5'
    ]
    target = 'Close'

    # Split into train and test sets
    train_size = int(len(data) * 0.8)
    train_data, test_data = data.iloc[:train_size], data.iloc[train_size:]

    # Standardize the data
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    X_train = feature_scaler.fit_transform(train_data[features])
    X_test = feature_scaler.transform(test_data[features])

    y_train = target_scaler.fit_transform(train_data[[target]])
    y_test = target_scaler.transform(test_data[[target]])

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # Create sequences
    X_train_seq, y_train_seq = create_sequences(X_train_tensor, y_train_tensor, seq_length)
    X_test_seq, y_test_seq = create_sequences(X_test_tensor, y_test_tensor, seq_length)

    # Create DataLoader
    train_dataset = TensorDataset(X_train_seq, y_train_seq)
    test_dataset = TensorDataset(X_test_seq, y_test_seq)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialize the model
    num_features = X_train_seq.shape[2]
    d_model = 64  # Project input features to 64 dimensions (must be divisible by num_heads)
    num_heads = 4

    model = TimeSeriesTransformer(num_features=num_features, num_heads=num_heads, num_layers=2, d_model=d_model)

    # Train the model
    train_model(model, train_loader, num_epochs=num_epochs)

    # Evaluate the model
    actuals, predictions = evaluate_model(model, test_loader, target_scaler)

    # Plot the results
    plot_predictions(actuals, predictions)
