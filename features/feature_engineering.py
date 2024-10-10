import ta
from transformers import pipeline

# Initialize BERT-based sentiment analysis
nlp = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')

def add_technical_indicators(data):
    """
    Add technical indicators like RSI, MACD, and Bollinger Bands to the dataset.
    """
    data['RSI'] = ta.momentum.RSIIndicator(close=data['Close']).rsi()
    macd = ta.trend.MACD(close=data['Close'])
    data['MACD'] = macd.macd_diff()
    bb = ta.volatility.BollingerBands(close=data['Close'])
    data['Bollinger_Width'] = bb.bollinger_wband()
    return data

def add_volume_indicators(data):
    """
    Add volume-based indicators like OBV, ADI, and CMF.
    """
    data['OBV'] = ta.volume.OnBalanceVolumeIndicator(close=data['Close'], volume=data['Volume']).on_balance_volume()
    data['ADI'] = ta.volume.AccDistIndexIndicator(high=data['High'], low=data['Low'], close=data['Close'], volume=data['Volume']).acc_dist_index()
    data['CMF'] = ta.volume.ChaikinMoneyFlowIndicator(high=data['High'], low=data['Low'], close=data['Close'], volume=data['Volume'], window=20).chaikin_money_flow()
    return data

def add_sentiment_analysis(data, stock_symbol, fetch_news_func):
    """
    Add sentiment analysis using BERT-based NLP to analyze stock-related news headlines.
    """
    headlines = fetch_news_func(stock_symbol)
    data['Headlines'] = headlines
    data['Sentiment'] = nlp(headlines)[0]['label']  
    return data

def add_lag_features(data):
    """
    Add lagged values of the closing price as features.
    """
    for lag in range(1, 6):
        data[f'Lag_{lag}'] = data['Close'].shift(lag)
    return data

def add_rolling_statistics(data):
    """
    Add rolling mean and rolling standard deviation to the dataset.
    """
    if len(data) > 20:
        data['Rolling_Mean_20'] = data['Close'].rolling(window=20).mean()
        data['Rolling_Std_20'] = data['Close'].rolling(window=20).std()
    else:
        data['Rolling_Mean_20'] = data['Close'].expanding(min_periods=1).mean()
        data['Rolling_Std_20'] = data['Close'].expanding(min_periods=1).std()
    return data

def add_moving_averages(data):
    """
    Add long-term moving averages (50-day, 100-day, 200-day) and detect crossovers.
    """
    data['MA_50'] = data['Close'].rolling(window=50).mean()
    data['MA_100'] = data['Close'].rolling(window=100).mean()
    data['MA_200'] = data['Close'].rolling(window=200).mean()

    # Detect moving average crossovers
    data['MA_Crossover'] = (data['MA_50'] > data['MA_200']).astype(int)  # 1 if 50-day MA crosses above 200-day MA
    return data

def preprocess_data(data, stock_symbol, fetch_news_func):
    """
    Perform data preprocessing by adding all features (technical indicators, sentiment analysis, etc.).
    """
    data = add_technical_indicators(data)
    data = add_volume_indicators(data)
    data = add_moving_averages(data)
    data = add_sentiment_analysis(data, stock_symbol, fetch_news_func)
    data = add_lag_features(data)
    data = add_rolling_statistics(data)
    
    data.dropna(inplace=True)  # Ensure no missing values remain
    return data
