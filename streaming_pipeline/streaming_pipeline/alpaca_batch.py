import datetime
import logging
import os
from typing import List

import requests
from bytewax.inputs import DynamicInput, StatelessSource

from streaming_pipeline import utils

logger = logging.getLogger(__name__)

class AlpacaNewsBatchInput(DynamicInput):
    def __init__(self, tickers: List[str], from_datetime: datetime.datetime, to_datetime: datetime.datetime):
        self.tickers = tickers
        self.from_datetime = from_datetime
        self.to_datetime = to_datetime

    def build(self, worker_index: int, worker_count: int):
        intervals = utils.split_time_range_into_intervals(self.from_datetime, self.to_datetime, worker_count)
        start, end = intervals[worker_index]
        logger.info(f"Worker {worker_index} processing from {start} to {end}")
        return AlpacaNewsBatchSource(self.tickers, start, end)

class AlpacaNewsBatchSource(StatelessSource):
    def __init__(self, tickers: List[str], from_datetime: datetime.datetime, to_datetime: datetime.datetime):
        self.client = build_alpaca_client(from_datetime, to_datetime, tickers)

    def next(self):
        news = self.client.list()
        if not news:
            raise StopIteration
        return news

    def close(self):
        logger.info("Closing AlpacaNewsBatchSource.")

def build_alpaca_client(from_datetime, to_datetime, tickers):
    api_key = os.getenv('ALPACA_API_KEY')
    api_secret = os.getenv('ALPACA_API_SECRET')
    if not api_key or not api_secret:
        raise KeyError("API key or API secret not provided.")
    return AlpacaNewsBatchClient(from_datetime, to_datetime, api_key, api_secret, tickers)

class AlpacaNewsBatchClient:
    NEWS_URL = "https://data.alpaca.markets/v1beta1/news"

    def __init__(self, from_datetime, to_datetime, api_key, api_secret, tickers):
        self.from_datetime = from_datetime
        self.to_datetime = to_datetime
        self.api_key = api_key
        self.api_secret = api_secret
        self.tickers = tickers

    def list(self):
        headers = {"Apca-Api-Key-Id": self.api_key, "Apca-Api-Secret-Key": self.api_secret}
        params = {
            "start": self.from_datetime.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end": self.to_datetime.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "tickers": ','.join(self.tickers),
            "limit": 50,
            "sort": "ASC"
        }
        response = requests.get(self.NEWS_URL, headers=headers, params=params)
        if response.status_code != 200:
            logger.error(f"Failed to fetch news data: {response.status_code}")
            return []
        return response.json().get("news", [])

