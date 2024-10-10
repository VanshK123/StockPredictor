import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

NEWSAPI_KEY = os.getenv('NEWSAPI_KEY')
