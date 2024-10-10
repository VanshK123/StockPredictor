import ta
from textblob import TextBlob

def add_technical_indicators(data):
    data['RSI'] = ta.momentum.RSIIndicator(close=data['Close']).rsi()
    macd = ta.trend.MACD(close=data['Close'])
    data['MACD'] = macd.macd_diff()
    bb = ta.volatility.BollingerBands(close=data['Close'])
    data['Bollinger_Width'] = bb.bollinger_wband()
    return data

def add_sentiment_analysis(data, stock_symbol, fetch_news_func):
    headlines = fetch_news_func(stock_symbol)
    data['Headlines'] = headlines
    data['Sentiment'] = TextBlob(headlines).sentiment.polarity
    return data

def add_lag_features(data):
    for lag in range(1, 6):
        data[f'Lag_{lag}'] = data['Close'].shift(lag)
    return data

def add_rolling_statistics(data):
    if len(data) > 20:
        data['Rolling_Mean_20'] = data['Close'].rolling(window=20).mean()
        data['Rolling_Std_20'] = data['Close'].rolling(window=20).std()
    else:
        data['Rolling_Mean_20'] = data['Close'].expanding(min_periods=1).mean()
        data['Rolling_Std_20'] = data['Close'].expanding(min_periods=1).std()
    return data

def preprocess_data(data, stock_symbol, fetch_news_func):
    data = add_technical_indicators(data)
    data = add_sentiment_analysis(data, stock_symbol, fetch_news_func)
    data = add_lag_features(data)
    data = add_rolling_statistics(data)
    data.dropna(inplace=True)  # Ensure no missing values remain
    return data
