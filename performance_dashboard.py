import pandas as pd
import matplotlib.pyplot as plt

class PerformanceDashboard:
    def __init__(self, history):
        self.df = pd.DataFrame(history)
        if not self.df.empty and "time" in self.df.columns:
            self.df["time"] = pd.to_datetime(self.df["time"])

    def trade_stats(self):
        sells = self.df[self.df["type"] == "SELL"] if not self.df.empty else self.df
        if sells is None or sells.empty:
            print("No completed trades yet.")
            return
        win_rate = (sells["pnl"] > 0).mean() * 100
        total_pnl = sells["pnl"].sum()
        avg_pnl = sells["pnl"].mean()
        mdd = self.max_drawdown()
        print("===== PERFORMANCE =====")
        print(f"Trades: {len(sells)}")
        print(f"Win rate: {win_rate:.2f}%")
        print(f"Total PnL: {total_pnl:.2f}")
        print(f"Avg PnL: {avg_pnl:.2f}")
        print(f"Max Drawdown: {mdd:.2f}%")

    def equity_curve(self):
        sells = self.df[self.df["type"] == "SELL"].copy() if not self.df.empty else self.df
        if sells is None or sells.empty:
            print("No equity data yet.")
            return
        sells = sells.sort_values("time")
        sells["equity"] = sells["balance"]
        plt.figure()
        plt.plot(sells["time"], sells["equity"])
        plt.title("Equity Curve")
        plt.xlabel("Time")
        plt.ylabel("Balance")
        plt.grid(True)
        plt.show()

    def max_drawdown(self):
        sells = self.df[self.df["type"] == "SELL"].copy() if not self.df.empty else self.df
        if sells is None or sells.empty:
            return 0.0
        equity = sells["balance"].values
        peak = equity[0]
        max_dd = 0.0
        for v in equity:
            peak = max(peak, v)
            dd = (peak - v) / peak * 100
            max_dd = max(max_dd, dd)
        return max_dd
