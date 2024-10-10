import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error

def train_model(model, train_loader, num_epochs=50, lr=0.001):
    """
    Train the given model using the provided data loader.
    """
    criterion = nn.MSELoss()  # Mean Squared Error loss for regression tasks
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

def evaluate_model(model, test_loader, target_scaler):
    """
    Evaluate the model on the test dataset and return actual values, predictions, and RMSE.
    """
    model.eval()
    predictions, actuals = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            output = model(X_batch)
            predictions.append(output.detach().numpy())
            actuals.append(y_batch.numpy())

    # Concatenate all predictions and actual values
    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)

    # Rescale predictions back to the original scale
    predictions = target_scaler.inverse_transform(predictions)
    actuals = target_scaler.inverse_transform(actuals)

    # Calculate RMSE
    rmse = mean_squared_error(actuals, predictions, squared=False)

    return actuals, predictions, rmse