from datetime import datetime
import sys 
sys.path.append("./")
from src.alpaca_news_api import download_historical_news

# January 2023
download_historical_news(
    from_date=datetime(2023, 1, 1),
    to_date=datetime(2023, 1, 5),
)
