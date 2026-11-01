"""
deep_learning.py
An LSTM neural network (PyTorch), only allowed for the LARGE tier (2+ years AND 1,000+ points).

How it works (same idea as our first LSTM, now inside the pipeline):
  1. Each period is described by a few numbers:
       - sales, scaled to "distance from the average" (average and spread from training data only)
       - weekday and day-of-year as sine/cosine pairs (so Sunday sits next to Monday,
         and 31 Dec next to 1 Jan)
       - festival facts for daily data: days until / since each selected festival (scaled 0-1)
  2. The LSTM reads the last 28 periods in order and keeps a "memory" as it goes.
     Its memory plus the calendar/festival facts of the NEXT period go into a small
     final layer that predicts that period's sales.
  3. To forecast 90 days, it predicts one day, adds that prediction to the history,
     and repeats ("recursive" forecast). Errors can build up, which is the main weakness.

Training safeguards:
  - The last horizon of the training data is kept aside for EARLY STOPPING: training stops
    when the error there has not improved for 20 rounds, and the best version is kept.
  - Fixed random seed, so results repeat.

Training is the slowest part of the whole pipeline (the backtest trains it 6 times).
"""

import numpy as np
import pandas as pd

from forecaster.data_check import HORIZON
from forecaster.models.base import ForecastModel
from forecaster.models.machine_learning import DAYS_SINCE_CAP, DAYS_UNTIL_CAP, festival_features

WINDOW = 28
MAX_EPOCHS = 200
PATIENCE = 20
BATCH = 64
SEED = 42


def known_features(dates: pd.DatetimeIndex, frequency: str, calendar) -> pd.DataFrame:
    """Facts known in advance for any date: calendar position and festival distances."""
    f = pd.DataFrame(index=dates)
    year_pos = dates.day_of_year.to_numpy() / 365.25
    f["year_sin"] = np.sin(2 * np.pi * year_pos)
    f["year_cos"] = np.cos(2 * np.pi * year_pos)
    if frequency == "daily":
        week_pos = dates.day_of_week.to_numpy() / 7
        f["week_sin"] = np.sin(2 * np.pi * week_pos)
        f["week_cos"] = np.cos(2 * np.pi * week_pos)
        if calendar is not None:
            fest = festival_features(dates, calendar)
            for col in fest.columns:   # scale to 0-1 so no input is much bigger than the others
                cap = DAYS_UNTIL_CAP if col.startswith("until_") else DAYS_SINCE_CAP
                f[col] = fest[col] / cap
    return f


class LSTMModel(ForecastModel):
    name = "lstm"
    family = "deep_learning"

    def __init__(self, frequency: str, calendar=None):
        super().__init__(frequency)
        self.calendar = calendar if (frequency == "daily" and calendar is not None and len(calendar)) else None
        self.horizon = HORIZON[frequency]

    def _fit(self, history):
        import torch
        from torch import nn

        torch.manual_seed(SEED)
        np.random.seed(SEED)

        self.mean = float(history.mean())
        self.std = float(history.std()) or 1.0
        known = known_features(history.index, self.frequency, self.calendar)
        self.known_cols = list(known.columns)
        scaled = ((history - self.mean) / self.std).to_numpy(dtype=np.float32)
        inputs = np.column_stack([scaled, known.to_numpy(dtype=np.float32)])   # one row per period
        next_known = known.to_numpy(dtype=np.float32)

        # Training examples: 28 periods in -> the next period's sales out.
        seqs = np.stack([inputs[i - WINDOW:i] for i in range(WINDOW, len(inputs))])
        cals = next_known[WINDOW:]
        ys = scaled[WINDOW:]
        n_val = min(self.horizon, len(ys) // 5)        # the most recent examples, for early stopping
        if len(ys) - n_val < 100:
            raise ValueError("Too little data to train an LSTM.")
        seqs, cals, ys = (torch.from_numpy(np.ascontiguousarray(a)) for a in (seqs, cals, ys))
        tr, va = slice(0, len(ys) - n_val), slice(len(ys) - n_val, len(ys))

        self.net = _Net(n_inputs=inputs.shape[1], n_known=next_known.shape[1])
        optimiser = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        loss_fn = nn.L1Loss()
        best, best_state, waited = float("inf"), None, 0
        n_train = len(ys) - n_val
        for _ in range(MAX_EPOCHS):
            self.net.train()
            order = torch.randperm(n_train)
            for start in range(0, n_train, BATCH):
                idx = order[start:start + BATCH]
                optimiser.zero_grad()
                loss = loss_fn(self.net(seqs[tr][idx], cals[tr][idx]), ys[tr][idx])
                loss.backward()
                optimiser.step()
            self.net.eval()
            with torch.no_grad():
                val = loss_fn(self.net(seqs[va], cals[va]), ys[va]).item()
            if val < best:
                best, waited = val, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                waited += 1
                if waited >= PATIENCE:
                    break
        self.net.load_state_dict(best_state)
        self.last_inputs = inputs[-WINDOW:].tolist()

    def _predict(self, n_steps):
        import torch

        future_known = known_features(self.future_dates(n_steps), self.frequency, self.calendar)
        rows = [list(r) for r in self.last_inputs]
        predictions = []
        self.net.eval()
        with torch.no_grad():
            for i in range(n_steps):
                known = future_known.iloc[i].to_numpy(dtype=np.float32)
                seq = torch.tensor(np.array([rows[-WINDOW:]], dtype=np.float32))
                cal = torch.tensor(known[None, :])
                pred = self.net(seq, cal).item()
                predictions.append(pred * self.std + self.mean)
                rows.append([pred] + known.tolist())      # the prediction becomes history
        return np.clip(np.array(predictions), 0, None)


def _make_net_class():
    from torch import nn
    import torch

    class Net(nn.Module):
        def __init__(self, n_inputs, n_known, hidden=32):
            super().__init__()
            self.lstm = nn.LSTM(input_size=n_inputs, hidden_size=hidden, batch_first=True)
            self.head = nn.Sequential(nn.Linear(hidden + n_known, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, seq, next_known):
            out, _ = self.lstm(seq)
            return self.head(torch.cat([out[:, -1, :], next_known], dim=1)).squeeze(1)

    return Net


def _Net(**kwargs):
    """Build the network. PyTorch is imported only when an LSTM is actually trained."""
    return _make_net_class()(**kwargs)


def deep_learning_models(frequency: str, calendar=None) -> list:
    return [LSTMModel(frequency, calendar)]
