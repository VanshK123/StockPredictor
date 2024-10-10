from data.fetch_data import fetch_stock_data, fetch_stock_news, get_sp500_companies
from features.feature_engineering import preprocess_data
from models.transformer_model import TimeSeriesTransformer
from models.train_model import train_model, evaluate_model
from utils.helpers import create_sequences, get_batch_size, plot_predictions
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

def main():
    # Ask the user how many S&P 500 companies they want to process
    try:
        num_companies = int(input("Enter the number of S&P 500 companies to process: "))
    except ValueError:
        print("Invalid input! Defaulting to 1 company.")
        num_companies = 1

    # Get the chosen number of S&P 500 companies
    stocks = get_sp500_companies(num_companies=num_companies)
    print(f"Selected stocks: {stocks}")
    
    start_date = '2010-01-01'
    end_date = '2023-10-01'
    seq_length = 60
    num_epochs = 10

    for stock_symbol in stocks:
        print(f"Processing {stock_symbol}...")

        # Fetch and preprocess data
        data = fetch_stock_data(stock_symbol, start_date, end_date)
        data = preprocess_data(data, stock_symbol, fetch_stock_news)

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

        # Get dynamic batch size
        batch_size = get_batch_size(len(data))
        print(f"Using batch size: {batch_size}")

        # Create DataLoader
        train_dataset = TensorDataset(torch.from_numpy(X_train_seq), torch.from_numpy(y_train_seq))
        test_dataset = TensorDataset(torch.from_numpy(X_test_seq), torch.from_numpy(y_test_seq))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        # Initialize the model
        num_features = X_train_seq.shape[2]
        d_model = 64
        num_heads = 4

        model = TimeSeriesTransformer(num_features=num_features, num_heads=num_heads, num_layers=2, d_model=d_model)

        # Train the model
        train_model(model, train_loader, num_epochs=num_epochs)

        # Evaluate the model
        actuals, predictions = evaluate_model(model, test_loader, target_scaler)

        # Plot the results for each stock
        plot_predictions(stock_symbol, actuals, predictions)

if __name__ == "__main__":
    main()
