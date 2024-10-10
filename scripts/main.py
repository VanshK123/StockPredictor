import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import numpy as np  # Required for combining features and predictions

# Import functions from your project
from data.fetch_data import fetch_stock_data, add_volume_indicators, add_bert_sentiment_analysis, add_sector_index, get_sp500_companies, fetch_stock_news
from features.feature_engineering import preprocess_data
from models.lstm_transformer_hybrid import LSTM_Transformer_Hybrid
from models.train_model import train_model, evaluate_model
from utils.helpers import create_sequences, get_batch_size, plot_predictions
from models.ensemble_model import ensemble_with_xgboost  # Import ensemble method

# Function to map sentiment strings to numeric values
def map_sentiment(sentiment):
    sentiment_mapping = {
        '1 star': 1,
        '2 stars': 2,
        '3 stars': 3,
        '4 stars': 4,
        '5 stars': 5
    }
    return sentiment_mapping.get(sentiment, 3)  # Default to '3 stars'

def main():
    # User input for the number of S&P 500 companies to process and epochs
    num_companies = int(input("Enter the number of S&P 500 companies to process: "))
    num_epochs = int(input("Enter the number of epochs for model training: "))

    # Fetch S&P 500 company symbols
    stocks = get_sp500_companies(num_companies=num_companies)
    print(f"Selected stocks: {stocks}")

    start_date = '2010-01-01'
    end_date = '2023-10-01'
    seq_length = 60  # Sequence length for LSTM/Transformer input

    # Process each stock individually
    for stock_symbol in stocks:
        print(f"Processing {stock_symbol}...")

        # Fetch stock data
        data = fetch_stock_data(stock_symbol, start_date, end_date)

        # Preprocess data (add technical indicators, sentiment analysis, etc.)
        data = preprocess_data(data, stock_symbol, fetch_stock_news)

        # Map sentiment to numeric values
        data['Sentiment'] = data['Sentiment'].apply(map_sentiment)

        # Features and target columns
        features = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            '10Y_Treasury_Yield', 'RSI', 'MACD', 'Bollinger_Width', 
            'Sentiment', 'Rolling_Mean_20', 'Rolling_Std_20',
            'Lag_1', 'Lag_2', 'Lag_3', 'Lag_4', 'Lag_5'
        ]
        target = 'Close'

        # Split into training and test sets (80% train, 20% test)
        train_size = int(len(data) * 0.8)
        train_data, test_data = data.iloc[:train_size], data.iloc[train_size:]

        # Scale features and target values
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

        # Create sequences for time series data
        def create_sequences(X, y, seq_length):
            sequences, targets = [], []
            for i in range(len(X) - seq_length):
                seq = X[i:i + seq_length]
                target = y[i + seq_length]
                sequences.append(seq)
                targets.append(target)
            return torch.stack(sequences), torch.stack(targets)

        X_train_seq, y_train_seq = create_sequences(X_train_tensor, y_train_tensor, seq_length)
        X_test_seq, y_test_seq = create_sequences(X_test_tensor, y_test_tensor, seq_length)

        # DataLoader for batching
        batch_size = 32
        train_dataset = TensorDataset(X_train_seq, y_train_seq)
        test_dataset = TensorDataset(X_test_seq, y_test_seq)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        # Initialize the LSTM-Transformer hybrid model
        num_features = X_train_seq.shape[2]  # Number of input features
        model = LSTM_Transformer_Hybrid(
            num_features=num_features,
            num_heads=4,
            num_layers=2,
            lstm_hidden_size=64,
            lstm_layers=2,
            d_model=128,
            dropout=0.3
        )

        # Train the model
        train_model(model, train_loader, num_epochs=num_epochs)

       # Evaluate the Transformer model
        actuals, transformer_predictions, rmse = evaluate_model(model, test_loader, target_scaler)
        print(f"RMSE for {stock_symbol}: {rmse}")

        # Reshape transformer predictions to match expected shape
        transformer_predictions = np.array(transformer_predictions).reshape(-1, 1)

        # Combine the original test features with Transformer predictions for the test set
        X_test_combined = np.hstack([X_test[:len(transformer_predictions)], transformer_predictions])

        # Add a placeholder column for the training set to match the shape (e.g., zeros)
        placeholder_predictions_train = np.zeros((X_train.shape[0], 1))  # Create a column of zeros
        X_train_combined = np.hstack([X_train, placeholder_predictions_train])

        # Train and evaluate XGBoost model
        xgboost_predictions = ensemble_with_xgboost(X_train_combined, y_train, X_test_combined)

        # Rescale XGBoost predictions back to the original scale
        xgboost_predictions_rescaled = target_scaler.inverse_transform(xgboost_predictions.reshape(-1, 1))

        # Output XGBoost predictions
        print(f"XGBoost predictions for {stock_symbol}: {xgboost_predictions_rescaled}")

        # Optionally, plot predictions vs actual
        plot_predictions(stock_symbol, actuals, xgboost_predictions_rescaled)



if __name__ == "__main__":
    main()
