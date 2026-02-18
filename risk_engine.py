from datetime import datetime, timedelta

class RiskEngine:
    def __init__(self, max_daily_loss_pct: float = 3.0, max_consecutive_losses: int = 5, cooldown_minutes: int = 60):
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_consecutive_losses = max_consecutive_losses
        self.cooldown_minutes = cooldown_minutes
        self.start_of_day_balance = None
        self.current_day = None
        self.consecutive_losses = 0
        self.cooldown_until = None

    def _reset_day(self, balance: float, now: datetime):
        self.start_of_day_balance = balance
        self.current_day = now.date()
        self.consecutive_losses = 0
        self.cooldown_until = None

    def _check_new_day(self, balance: float, now: datetime):
        if self.current_day != now.date():
            self._reset_day(balance, now)

    def daily_loss_pct(self, balance: float) -> float:
        if self.start_of_day_balance is None:
            return 0.0
        return (self.start_of_day_balance - balance) / self.start_of_day_balance * 100

    def daily_loss_exceeded(self, balance: float) -> bool:
        return self.daily_loss_pct(balance) >= self.max_daily_loss_pct

    def record_trade_result(self, pnl: float, now: datetime):
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        if self.consecutive_losses >= self.max_consecutive_losses:
            self.cooldown_until = now + timedelta(minutes=self.cooldown_minutes)

    def in_cooldown(self, now: datetime) -> bool:
        return self.cooldown_until is not None and now < self.cooldown_until

    def can_trade(self, balance: float, now: datetime) -> bool:
        if self.current_day is None:
            self._reset_day(balance, now)
        self._check_new_day(balance, now)
        if self.daily_loss_exceeded(balance):
            return False
        if self.in_cooldown(now):
            return False
        return True
