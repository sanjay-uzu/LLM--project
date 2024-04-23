import json
import logging
import os
from typing import List, Union

import websocket
from bytewax.inputs import DynamicInput, StatelessSource

logger = logging.getLogger(__name__)

class AlpacaNewsStreamInput(DynamicInput):
    """Input class to receive streaming news data from Alpaca's real-time API.

    Attributes:
        tickers (List[str]): List of stock tickers to subscribe to, use "*" for all.
    """
    def __init__(self, tickers: List[str]):
        self.tickers = tickers

    def build(self, worker_index: int, worker_count: int):
        tickers_per_worker = len(self.tickers) // worker_count
        allocated_tickers = self.tickers[worker_index * tickers_per_worker: (worker_index + 1) * tickers_per_worker]
        logger.info(f"Worker {worker_index} allocated tickers: {allocated_tickers}")
        return AlpacaNewsStreamSource(allocated_tickers)

class AlpacaNewsStreamSource(StatelessSource):
    """A source for streaming news data from Alpaca API.

    Attributes:
        tickers (List[str]): Ticker symbols to subscribe to.
    """
    def __init__(self, tickers: List[str]):
        self.client = build_alpaca_client(tickers=tickers)
        self.client.start()
        self.client.subscribe()

    def next(self):
        return self.client.recv()

    def close(self):
        self.client.unsubscribe()
        self.client.close()

def build_alpaca_client(tickers: List[str]):
    api_key = os.getenv('ALPACA_API_KEY')
    api_secret = os.getenv('ALPACA_API_SECRET')
    if not api_key or not api_secret:
        raise KeyError("API key or API secret not set in the environment variables.")
    return AlpacaNewsStreamClient(api_key, api_secret, tickers)

class AlpacaNewsStreamClient:
    """Client that uses a websocket to stream news data from Alpaca.

    Attributes:
        api_key (str): Alpaca API key.
        api_secret (str): Alpaca API secret.
        tickers (List[str]): Tickers to subscribe to.
        ws (websocket.WebSocket): Websocket connection.
    """
    NEWS_URL = "wss://stream.data.alpaca.markets/v1beta1/news"

    def __init__(self, api_key: str, api_secret: str, tickers: List[str]):
        self.api_key = api_key
        self.api_secret = api_secret
        self.tickers = tickers
        self.ws = None

    def start(self):
        self.ws = websocket.create_connection(self.NEWS_URL)
        self.authenticate()

    def authenticate(self):
        message = {
            "action": "authenticate",
            "data": {
                "key_id": self.api_key,
                "secret_key": self.api_secret
            }
        }
        self.ws.send(json.dumps(message))
        response = json.loads(self.ws.recv())
        if response.get('stream', '') == 'authentication' and response.get('data', {}).get('status', '') == 'authorized':
            logger.info("Successfully authenticated.")
        else:
            logger.error("Failed to authenticate with Alpaca stream.")
            raise ConnectionError("Authentication failed.")

    def subscribe(self):
        message = {
            "action": "listen",
            "data": {
                "streams": [f"news.{ticker}" for ticker in self.tickers]
            }
        }
        self.ws.send(json.dumps(message))
        logger.info("Subscribed to news streams for tickers.")

    def recv(self):
        return json.loads(self.ws.recv())

    def unsubscribe(self):
        message = {
            "action": "unlisten",
            "data": {
                "streams": [f"news.{ticker}" for ticker in self.tickers]
            }
        }
        self.ws.send(json.dumps(message))
        logger.info("Unsubscribed from news streams.")

    def close(self):
        self.ws.close()
        logger.info("Websocket connection closed.")
