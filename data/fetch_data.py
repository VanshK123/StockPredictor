import yfinance as yf
import pandas_datareader.data as web
import pandas as pd 
from dotenv import load_dotenv
from newsapi import NewsApiClient
import os
import random

# Load environment variables
load_dotenv()

# Initialize NewsAPI
newsapi_key = os.getenv('NEWSAPI_KEY')
newsapi = NewsApiClient(api_key=newsapi_key)

def fetch_stock_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    treasury_yield = web.DataReader('DGS10', 'fred', start_date, end_date)
    treasury_yield = treasury_yield.resample('D').ffill()
    data = data.merge(treasury_yield, left_index=True, right_index=True, how='left')
    data.rename(columns={'DGS10': '10Y_Treasury_Yield'}, inplace=True)
    return data

def fetch_stock_news(stock_symbol):
    news = newsapi.get_everything(q=stock_symbol, language='en', sort_by='relevancy', page_size=5)
    headlines = [article['title'] for article in news['articles']]
    return ' '.join(headlines)

def get_sp500_companies(num_companies=5):
    """Fetch the S&P 500 companies from Wikipedia and randomly select a subset."""
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    sp500_table = pd.read_html(url, header=0)[0]
    sp500_companies = sp500_table['Symbol'].tolist()
    return random.sample(sp500_companies, num_companies)
