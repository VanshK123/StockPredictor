import matplotlib.pyplot as plt
import numpy as np

def create_sequences(X, y, seq_length=60):
    sequences = []
    targets = []
    for i in range(len(X) - seq_length):
        seq = X[i:i + seq_length]
        target = y[i + seq_length]
        sequences.append(seq)
        targets.append(target)
    return np.stack(sequences), np.stack(targets)

def get_batch_size(data_length):
    if data_length < 1000:
        return 16
    elif data_length < 5000:
        return 32
    else:
        return 64

def plot_predictions(stock_symbol, actuals, predictions):
    plt.figure(figsize=(12, 6))
    plt.plot(actuals, label='Actual Prices')
    plt.plot(predictions, label='Predicted Prices')
    plt.legend()
    plt.title(f'Actual vs Predicted Stock Prices for {stock_symbol}')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.savefig(f'predicted_vs_actual_{stock_symbol}.png')
    plt.close()
    print(f"Plot saved as 'predicted_vs_actual_{stock_symbol}.png'")
