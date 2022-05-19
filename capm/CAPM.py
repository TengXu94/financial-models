"""
The CAPM model:

expected_stock_return = r_f + \beta_i (expected_market_return - r_f)

r_f: risk free return
beta_i: stock i idiosyncratic risk
"""
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)


@dataclass
class CAPM:
    def __init__(self):
        self.logger = logging.getLogger("CAPM")
        self.market = self.load_data("data/SPY.csv")
        self.market = self.normalize(self.market, "close")
        self.market = self.compute_returns(self.market, "close")
        self.rf = 0

    def load_data(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        df = df[(df.time > "2012-01-11") & (df.time < "2020-08-12")]
        df = df.sort_values("time")
        return df

    def normalize(self, df: pd.DataFrame, column: str) -> pd.DataFrame:

        df[column] = df[column] / df.iloc[0].close

        return df

    def compute_returns(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        df["returns"] = ((df[column] - df[column].shift(1)) / df[column].shift(1)) * 100

        df = df[~df["returns"].isna()]
        return df

    def compute_expected_return(self, stock: pd.DataFrame) -> pd.DataFrame:
        stock = self.normalize(stock, "close")
        stock = self.compute_returns(stock, "close")
        beta, alpha = np.polyfit(self.market.returns, stock.returns, 1)
        self.logger.info("Beta for is = {} and alpha is = {}".format(beta, alpha))

        market_return = self.market.returns.mean() * 252
        er_aapl = self.rf + (beta * (market_return - self.rf))
        self.logger.info("Stock expected return {}".format(er_aapl))


if __name__ == "__main__":
    capm = CAPM()
    aapl = capm.load_data("data/AAPL.csv")
    capm.compute_expected_return(aapl)
