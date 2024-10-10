import yfinance as yf
import pandas_datareader.data as web
import pandas as pd 
from dotenv import load_dotenv
from newsapi import NewsApiClient
from transformers import pipeline 
import ta  
import os
import random

# Load environment variables
load_dotenv()

# Initialize NewsAPI
newsapi_key = os.getenv('NEWSAPI_KEY')
newsapi = NewsApiClient(api_key=newsapi_key)

# Initialize BERT-based sentiment analysis
nlp = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')

def fetch_stock_data(symbol, start_date, end_date):
    """
    Fetch stock data and 10Y Treasury yield, then merge them into one DataFrame.
    """
    data = yf.download(symbol, start=start_date, end=end_date)
    treasury_yield = web.DataReader('DGS10', 'fred', start_date, end_date)
    treasury_yield = treasury_yield.resample('D').ffill()  # Fill missing dates
    data = data.merge(treasury_yield, left_index=True, right_index=True, how='left')
    data.rename(columns={'DGS10': '10Y_Treasury_Yield'}, inplace=True)
    return data

def fetch_stock_news(stock_symbol):
    """
    Fetch news headlines related to a stock symbol using the NewsAPI.
    """
    news = newsapi.get_everything(q=stock_symbol, language='en', sort_by='relevancy', page_size=5)
    headlines = [article['title'] for article in news['articles']]
    return ' '.join(headlines)

def add_volume_indicators(data):
    """
    Add volume-based indicators like OBV, ADI, and CMF.
    """
    # On-Balance Volume (OBV)
    data['OBV'] = ta.volume.OnBalanceVolumeIndicator(close=data['Close'], volume=data['Volume']).on_balance_volume()
    
    # Accumulation/Distribution Index (ADI)
    data['ADI'] = ta.volume.AccDistIndexIndicator(high=data['High'], low=data['Low'], close=data['Close'], volume=data['Volume']).acc_dist_index()

    # Chaikin Money Flow (CMF)
    data['CMF'] = ta.volume.ChaikinMoneyFlowIndicator(high=data['High'], low=data['Low'], close=data['Close'], volume=data['Volume'], window=20).chaikin_money_flow()

    return data

def add_bert_sentiment_analysis(data):
    """
    Add sentiment analysis to the dataset using BERT model to analyze stock news headlines.
    """
    # Assuming news headlines are fetched and stored in the 'Headlines' column
    data['Sentiment'] = data['Headlines'].apply(lambda x: nlp(x)[0]['label'])
    return data

def add_sector_index(data, sector_symbol):
    """
    Add sector or industry index data (e.g., S&P 500) to the dataset.
    """
    sector_data = yf.download(sector_symbol, start=data.index.min(), end=data.index.max())
    data = data.merge(sector_data[['Close']], left_index=True, right_index=True, how='left')
    data.rename(columns={'Close': f'{sector_symbol}_Close'}, inplace=True)
    return data

def get_sp500_companies(num_companies=5):
    """
    Fetch the S&P 500 companies from Wikipedia and randomly select a subset.
    """
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    sp500_table = pd.read_html(url, header=0)[0]
    sp500_companies = sp500_table['Symbol'].tolist()
    return random.sample(sp500_companies, num_companies)
