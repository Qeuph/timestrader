import json
import os
import pandas as pd
from datetime import datetime

class PortfolioManager:
    def __init__(self, filename="portfolio.json"):
        self.filename = filename
        self.trades = self._load_trades()

    def _load_trades(self):
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []

    def save_trade(self, trade_data):
        """
        Save a new trade to the portfolio.
        """
        trade_entry = trade_data.copy()
        # Convert any timestamps or non-serializable objects
        if 'date' in trade_entry and hasattr(trade_entry['date'], 'isoformat'):
            trade_entry['date'] = trade_entry['date'].isoformat()
        elif 'date' not in trade_entry:
            trade_entry['date'] = datetime.now().isoformat()

        self.trades.append(trade_entry)

        with open(self.filename, 'w') as f:
            json.dump(self.trades, f, indent=4)

    def get_trades_df(self):
        if not self.trades:
            return pd.DataFrame()
        df = pd.DataFrame(self.trades)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        return df

    def clear_portfolio(self):
        self.trades = []
        if os.path.exists(self.filename):
            os.remove(self.filename)

    def get_summary(self):
        df = self.get_trades_df()
        if df.empty:
            return {"total_trades": 0, "total_pnl": 0}

        # This is a simple summary of saved trades
        return {
            "total_trades": len(df),
            "total_signals": df['signal'].value_counts().to_dict()
        }
