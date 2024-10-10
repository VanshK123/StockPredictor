
# Stock Price Prediction Using Transformer

---

## Project Overview

This project predicts stock prices for multiple S&P 500 companies using a **Transformer-based model** designed for time series data. The model is augmented with **feature engineering**, including **technical indicators** (RSI, MACD, Bollinger Bands), **sentiment analysis** (from financial news), and **lag features** (previous stock prices). The system supports dynamic user input to choose how many S&P 500 companies to analyze.

---

## Project Structure

```
stock_prediction_project/
│
├── data/
│   └── __init__.py
│   └── fetch_data.py         # Fetch stock data, news, and S&P 500 companies
│
├── features/
│   └── __init__.py
│   └── feature_engineering.py # Feature engineering for technical indicators and sentiment analysis
│
├── models/
│   └── __init__.py
│   └── transformer_model.py   # Transformer-based model for time series prediction
│   └── train_model.py         # Functions to train and evaluate the model
│
├── utils/
│   └── __init__.py
│   └── helpers.py             # Helper functions for batch size, sequence generation, and plotting
│
├── config/
│   └── __init__.py
│   └── config.py              # Configurations like API keys
│
├── scripts/
│   └── main.py                # Main entry point to run the project
│
├── .env                       # Environment variables (API keys)
├── .gitignore                 # Ignore unnecessary files (including .env)
└── requirements.txt           # Dependencies
```

---

## Setup Instructions

Follow these steps to set up and run the project on your local machine.

### 1. Clone the Repository

```bash
git clone https://github.com/VanshK123/StockPredictor.git
cd StockPredictor
```

### 2. Set Up a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # For Linux/macOS
venv\Scripts\activate  # For Windows
```

### 3. Install Required Dependencies

Install all the dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

**Dependencies:**
- `yfinance`
- `pandas`
- `numpy`
- `torch`
- `matplotlib`
- `scikit-learn`
- `ta`
- `newsapi-python`
- `textblob`
- `python-dotenv`

### 4. Configure Environment Variables

1. Create a `.env` file in the project root directory:
   ```bash
   touch .env
   ```

2. Add your `NewsAPI` key to the `.env` file:
   ```bash
   NEWSAPI_KEY=your_news_api_key
   ```

3. Add `.env` to `.gitignore` to avoid committing it to version control.

### 5. Configure `PYTHONPATH` (Optional)

To ensure smooth imports, set the project root as `PYTHONPATH`:

- **Temporarily**:
  ```bash
  PYTHONPATH=$(pwd) python scripts/main.py
  ```

- **Permanently**:
  Add the following to your `.bashrc` or `.zshrc`:
  ```bash
  export PYTHONPATH="/path/to/stock_prediction_project"
  ```

---

## How to Run the Project

1. Activate the virtual environment:

```bash
source venv/bin/activate
```

2. Run the main script:

```bash
python scripts/main.py
```

3. **Input the Number of Companies**: The program will prompt you to enter how many S&P 500 companies you want to process (e.g., 3):

```bash
Enter the number of S&P 500 companies to process: 3
```
4. **Input the Number of Epochs**: You will also be prompted to enter the number of epochs to use for training the model (e.g., 20):

```bash
Enter the number of epochs for model training: 20
```


The program will then fetch stock data for the selected companies, preprocess it, train a Transformer-based model, and output the stock price predictions along with plots showing actual vs. predicted prices.

---

## Code Explanation

### 1. Data Collection (from `fetch_data.py`)

- **Stock Data**: Uses `yfinance` to fetch historical stock prices.
- **S&P 500 List**: Scrapes S&P 500 companies from Wikipedia.
- **News Headlines**: Uses `NewsAPI` to fetch real-world news headlines for sentiment analysis.

### 2. Feature Engineering (from `feature_engineering.py`)

- Adds **technical indicators** like RSI, MACD, and Bollinger Bands.
- Performs **sentiment analysis** using **TextBlob**.
- Adds **lag features** and **rolling statistics** to enhance predictive power.

### 3. Transformer Model (from `transformer_model.py`)

- Implements a **Transformer** architecture with `num_layers`, `num_heads`, and an `nn.Linear` decoder for stock price prediction.
- Processes time-series stock data with attention mechanisms to learn dependencies across time.

### 4. Model Training and Evaluation (from `train_model.py`)

- Trains the model using the **MSE loss** function and **Adam optimizer**.
- Evaluates the model by comparing predicted stock prices to actual prices, and generates a plot of the results.

### 5. Utilities (from `helpers.py`)

- Helper functions for **batch size calculation**, **sequence generation** for time series data, and **plotting results**.

---

## How It Works

1. **Fetch S&P 500 Companies**: User chooses the number of companies from the S&P 500 list.
2. **Fetch Stock Data and News**: The system fetches historical stock prices and related news headlines.
3. **Preprocess the Data**: Includes feature engineering (technical indicators, sentiment analysis, etc.).
4. **Train the Model**: A Transformer-based model is trained on the time-series stock data.
5. **Make Predictions**: The model predicts stock prices, and results are plotted against actual prices.

---

## Next Steps and Improvements

- **Model Tuning**: Add hyperparameter tuning for the Transformer model.
- **Sentiment Analysis**: Replace the basic sentiment analysis with a more sophisticated NLP model (e.g., BERT).
- **Additional Features**: Integrate other data sources like macroeconomic indicators or alternative data (e.g., social media sentiment).

---

## Conclusion

This project uses advanced AI techniques, such as **Transformers** and **time series forecasting**, combined with real-world data, including stock prices and news sentiment. It serves as an excellent educational tool for understanding financial forecasting and AI-driven predictions.