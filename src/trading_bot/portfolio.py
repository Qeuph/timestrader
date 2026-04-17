import json
import os
import pandas as pd
import tempfile
import uuid
from datetime import datetime, timezone

class PortfolioManager:
    def __init__(self, filename="portfolio.json"):
        self.filename = filename
        self.schema_version = 2
        self.trades = self._load_trades()

    def _load_trades(self):
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r') as f:
                    trades = json.load(f)
                    return [self._migrate_trade(t) for t in trades]
            except (json.JSONDecodeError, OSError):
                return []
        return []

    def save_trade(self, trade_data):
        """
        Save a new trade to the portfolio.
        """
        trade_entry = trade_data.copy()
        # Convert any timestamps or non-serializable objects.
        if 'date' in trade_entry and hasattr(trade_entry['date'], 'isoformat'):
            trade_entry['date'] = trade_entry['date'].isoformat()
        elif 'date' not in trade_entry:
            trade_entry['date'] = datetime.now(timezone.utc).isoformat()

        trade_entry = self._migrate_trade(trade_entry)
        self._validate_trade(trade_entry)

        self.trades.append(trade_entry)
        self._write_atomic(self.trades)

    def _write_atomic(self, payload):
        directory = os.path.dirname(os.path.abspath(self.filename)) or "."
        with tempfile.NamedTemporaryFile("w", delete=False, dir=directory, suffix=".tmp") as tmp:
            json.dump(payload, tmp, indent=4)
            tmp.flush()
            os.fsync(tmp.fileno())
            temp_name = tmp.name
        os.replace(temp_name, self.filename)

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

    def _migrate_trade(self, trade_entry):
        migrated = trade_entry.copy()
        migrated.setdefault("schema_version", self.schema_version)
        migrated.setdefault("trade_id", str(uuid.uuid4()))
        migrated.setdefault("created_at_utc", datetime.now(timezone.utc).isoformat())
        migrated.setdefault("model_repo_id", None)
        migrated.setdefault("period", None)
        migrated.setdefault("interval", None)
        migrated.setdefault("horizon", None)
        migrated.setdefault("selected_indicators", [])
        migrated.setdefault("strategy_config", {})
        return migrated

    def _validate_trade(self, trade_entry):
        required_fields = [
            "trade_id",
            "created_at_utc",
            "ticker",
            "signal",
            "model_repo_id",
            "period",
            "interval",
            "horizon",
            "selected_indicators",
            "strategy_config",
        ]
        missing = [field for field in required_fields if field not in trade_entry]
        if missing:
            raise ValueError(f"Trade payload missing required fields: {missing}")
