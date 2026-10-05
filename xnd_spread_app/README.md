# XND put spread planner

A Streamlit app for planning put credit spreads on **XND**, the Nasdaq-100 micro index (one hundredth of the
Nasdaq-100, cash-settled, taxed as an index option). It is a planning tool, not financial advice.

For this week and for the month ahead it shows:

1. **How far the index could move.** What the option prices imply, next to what a model built on 25 years of
   history expects, with the model's error bar. QQQ follows the same index, so the move in percent is QQQ's too.
2. **The spread with the best chance of collecting a premium you choose**, inside a loss cap you choose, with the
   alternatives listed safest first.
3. **A recommended trade.** Sell the 20-delta put and buy the 10-delta put, sized to your loss cap, with the model's
   view of whether the premium is rich, fair or thin for the move it expects. The monthly spread also follows the
   200-day rule: no new spread when the index closed below its 200-day average.

## Run it

Needs Python 3.12 or newer (the ThetaData client requires it).

```
pip install -r requirements.txt
streamlit run app.py
```

With no key it starts on **demo quotes**: made-up option prices built from the real index level and VXN, so you can
try every screen.

### Your ThetaData key

The key goes in Streamlit Secrets and nowhere else. It is never written in the code.

- **On your own computer:** copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and put the key in it.
  `.gitignore` keeps that file out of git.
- **On Streamlit Community Cloud:** App settings → Secrets, then paste `THETADATA_API_KEY = "..."`.
  In Advanced settings pick Python 3.12 or newer. If this folder sits inside a larger repo, set the main file
  path to `xnd_spread_app/app.py`.

If you put this in a public repo, check before every push that `secrets.toml` is not in the commit.

## Where the numbers come from

| What | Source |
|---|---|
| XND option quotes | ThetaData. With the market open: a live snapshot if your plan gives one, otherwise the latest half-hour quotes on record. With the market closed: the last session's final full half-hour, because the very last quotes of the day have the widest gaps between bid and ask. The app always shows the time of the quotes it is using. |
| Nasdaq-100, VXN and QQQ daily closes | `history.csv` (saved with the app, 2001 to October 2026), topped up from Yahoo Finance, or FRED if Yahoo doesn't answer. |
| T-bill rate | FRED (4% if it doesn't answer). |

## How to read it

- **Options imply ±X%**: the one-standard-deviation move priced into at-the-money options.
- **The model expects ±Y% (usually ±a% to ±b%)**: the model's estimate of the same thing, and the range the actual
  size of the moves has landed in 8 times out of 10. The model predicts how big the move will be, from VXN and
  from how much the index moved over the last 5, 20 and 60 days. It does not predict direction.
- **Chance you keep it all**: the chance the index finishes above the strike you sell. The model's figure is a count
  over 20 years of comparable moves, with a range. The options' figure is what their prices imply, including the
  fact that lower strikes trade at higher volatility (which moves a 20-delta put's chance by several points).
- **A target your cap can't reach**: the app says "not possible inside your cap", then shows the smallest share of the
  account you would have to put at risk to reach it, the exact spread that does it, and the other ways to get there
  with what each puts at risk. Raise the loss cap to that figure and the target becomes possible.
- **Model's value**: what a spread would have made per dollar at risk, on average, across the model's history of
  comparable moves, after costs. Around zero is normal. It is an estimate from past moves, not a promise.

## What it can't do

- It can't tell you which way the market will go.
- It doesn't know what you already hold. The loss cap is for everything open, so don't add a second spread on top
  of one that is still on.
- Holidays aren't in its calendar: in a holiday week it counts one trading session too many.
- It doesn't show backtest results. Those live in the project's notebooks.

## Files

- `app.py`: the screens.
- `core.py`: option maths, the spread finder, the range model, the rule. No network, no Streamlit.
- `data.py`: ThetaData, price history, demo quotes.
- `history.csv`: daily closes saved with the app.
- `tests/`: `python tests/test_core.py`, `python tests/test_data.py`, `python tests/test_app.py`. None needs a key
  or a network connection.
