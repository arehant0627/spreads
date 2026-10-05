"""Data for the XND put spread planner.

  - option quotes for XND (the Nasdaq-100 micro index) from ThetaData,
  - daily closes of the Nasdaq-100, VXN and QQQ: the copy saved with the app (history.csv), topped up from Yahoo Finance
    or FRED when they answer,
  - a demo market with the same shapes, for trying the app without a ThetaData key.

The ThetaData key is never read here. app.py takes it from Streamlit Secrets and passes it to make_client()."""
from __future__ import annotations

import datetime as dt
import io
import math
import os
import time

import numpy as np
import pandas as pd

SYMBOL = "XND"
NY = "America/New_York"
HERE = os.path.dirname(os.path.abspath(__file__))
SAVED_HISTORY = os.path.join(HERE, "history.csv")


def now_ny() -> pd.Timestamp:
    return pd.Timestamp.now(tz=NY)


def ny_time(x) -> pd.Series:
    """Timestamps as New York time, whether they arrive with a time zone or as plain New York clock times."""
    x = pd.Series(x).reset_index(drop=True)
    if not len(x):
        return pd.to_datetime(x).dt.tz_localize(NY)
    first = x.iloc[0]
    if isinstance(first, str):
        aware = first.endswith("Z") or "+" in first[10:] or "-" in first[10:]
    else:
        aware = getattr(first, "tzinfo", None) is not None
    if aware:
        return pd.to_datetime(x, utc=True).dt.tz_convert(NY)
    return pd.to_datetime(x).dt.tz_localize(NY)


# ---------------------------------------------------------------- ThetaData
def make_client(api_key: str):
    from thetadata import ThetaClient
    return ThetaClient(api_key=api_key, dataframe_type="pandas")


def _call(fn, tries=3, **kw):
    """Call a ThetaData method; 'no data' comes back as an empty frame, other errors retry."""
    delay = 1.5
    for attempt in range(tries):
        try:
            out = fn(**kw)
            return out if out is not None else pd.DataFrame()
        except Exception as e:                                  # noqa: BLE001
            if "no data" in str(e).lower() or type(e).__name__ == "NoDataFoundError":
                return pd.DataFrame()
            if attempt == tries - 1:
                raise
            time.sleep(delay)
            delay *= 2


def list_expirations(client, symbol=SYMBOL, today=None) -> list[dt.date]:
    """Every listed expiration from today on."""
    df = _call(client.option_list_expirations, symbol=symbol)
    if df.empty:
        return []
    col = next((c for c in df.columns if "exp" in str(c).lower()), df.columns[-1])
    d = pd.to_datetime(df[col].astype(str), errors="coerce").dropna().dt.date
    today = today or now_ny().date()
    return sorted({x for x in d if x >= today})


def _norm(df: pd.DataFrame, expiration: dt.date):
    """Standard columns, only the requested expiration (a response that mixes expirations would scramble the chain),
    quotes with a price on at least one side, and a note of what came back for the app's data check."""
    df = df.rename(columns=lambda c: str(c).lower())
    diag = {"rows returned": int(len(df))}
    if "expiration" in df.columns:
        e = pd.to_datetime(df["expiration"].astype(str), errors="coerce").dt.date
        diag["expirations returned"] = int(e.nunique())
        df = df[e == expiration]
    keep = [c for c in ["strike", "right", "bid", "ask", "timestamp"] if c in df.columns]
    q = df[keep].copy()
    if {"bid", "ask"} <= set(q.columns):
        q = q[(pd.to_numeric(q["bid"], errors="coerce") > 0) | (pd.to_numeric(q["ask"], errors="coerce") > 0)]
    q = q.reset_index(drop=True)
    if "timestamp" in q.columns and len(q):
        q["timestamp"] = ny_time(q["timestamp"])
    diag["rows used"] = int(len(q))
    return q, diag


def recent_sessions(now=None, n=5) -> list[dt.date]:
    """Today (if it is a weekday and the market has opened) and the weekdays before it, newest first."""
    now = now or now_ny()
    d = now.date()
    if now.weekday() >= 5 or now.hour * 60 + now.minute < 9 * 60 + 45:
        d -= dt.timedelta(days=1)
    out = []
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d -= dt.timedelta(days=1)
    return out


def market_open(now=None) -> bool:
    """Weekdays 9:30 am to 4:15 pm New York time (index options trade 15 minutes past the stock close). Holidays not known."""
    now = now or now_ny()
    return now.weekday() < 5 and 9 * 60 + 30 <= now.hour * 60 + now.minute < 16 * 60 + 15


def fetch_chain(client, expiration: dt.date, symbol=SYMBOL, now=None):
    """One expiration's quotes: (quotes, source, time of the quotes, notes).

    Market open: the live snapshot, or the latest half-hour quotes on record today if the plan gives no snapshot.
    Market closed: the last session's final full half-hour of quotes. A snapshot taken after the close holds the very
    last quotes of the day, when the gaps between bid and ask are at their widest, so it is only the fallback."""
    now = now or now_ny()
    notes, held = {}, None
    try:
        snap = _call(client.option_snapshot_quote, symbol=symbol, expiration=expiration)
    except Exception as e:                                      # noqa: BLE001  (a plan without snapshots raises here)
        snap, notes["snapshot"] = pd.DataFrame(), f"{type(e).__name__}: {str(e)[:120]}"
    if len(snap):
        q, diag = _norm(snap, expiration)
        if len(q):
            ts = q["timestamp"].max() if "timestamp" in q.columns else now
            if market_open(now) and (now - ts).total_seconds() < 45 * 60:
                return q, "live", ts, dict(notes, **diag)
            held = (q, "after hours", ts, dict(notes, **diag))
    for day in recent_sessions(now):
        try:
            h = _call(client.option_history_quote, symbol=symbol, expiration=expiration, interval="30m", date=day)
        except Exception as e:                                  # noqa: BLE001
            notes["history"] = f"{type(e).__name__}: {str(e)[:120]}"
            break
        if not len(h):
            continue
        q, diag = _norm(h, expiration)
        if not len(q) or "timestamp" not in q.columns:
            continue
        bars = q.groupby("timestamp").size()
        latest = bars[bars >= 0.5 * bars.max()].index.max()     # the newest half-hour with a full set of quotes
        q = q[q["timestamp"] == latest].drop_duplicates(["strike", "right"], keep="last").reset_index(drop=True)
        if held is not None and latest.date() < held[2].date():
            break                                               # the snapshot is from a later session than this history
        notes.update(diag)
        notes["rows used"] = int(len(q))
        return q, ("earlier today" if day == now.date() and market_open(now) else "last close"), latest, notes
    if held is not None:
        return held
    return pd.DataFrame(), "none", None, notes


# ---------------------------------------------------------------- prices
def saved_history() -> pd.DataFrame:
    h = pd.read_csv(SAVED_HISTORY, parse_dates=["date"]).set_index("date").sort_index()
    return h[["ndx", "vxn", "qqq"]]


def _yahoo(start: dt.date) -> pd.DataFrame:
    import yfinance as yf
    px = yf.download(["^NDX", "^VXN", "QQQ"], start=str(start), auto_adjust=False, progress=False, threads=False)
    c = px["Close"].rename(columns={"^NDX": "ndx", "^VXN": "vxn", "QQQ": "qqq"})
    c.index = pd.to_datetime(c.index).tz_localize(None)
    return c[["ndx", "vxn", "qqq"]].dropna(subset=["ndx"])


def _fred(start: dt.date) -> pd.DataFrame:
    import requests
    out = {}
    for name, code in (("ndx", "NASDAQ100"), ("vxn", "VXNCLS")):
        r = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}&cosd={start}", timeout=20)
        r.raise_for_status()
        raw = pd.read_csv(io.StringIO(r.text))
        out[name] = pd.Series(pd.to_numeric(raw.iloc[:, 1], errors="coerce").to_numpy(), index=pd.to_datetime(raw.iloc[:, 0]))
    c = pd.DataFrame(out).dropna(subset=["ndx"])
    c["qqq"] = np.nan
    return c


def load_history(now=None, sources=None) -> tuple[pd.DataFrame, str]:
    """Daily closes of the Nasdaq-100 (ndx), VXN and QQQ since 2001, finished days only: (table, where it came from).

    Starts from the copy saved with the app and adds the days since from Yahoo Finance, or FRED if Yahoo doesn't
    answer. If neither answers the saved copy is used as it is, and the app shows how old it is."""
    now = now or now_ny()
    h = saved_history()
    note = "saved copy only"
    start = (h.index[-1] - pd.Timedelta(days=10)).date()
    for name, fn in (sources or (("Yahoo Finance", _yahoo), ("FRED", _fred))):
        try:
            new = fn(start)
            new = new[new.index > h.index[-1] - pd.Timedelta(days=10)]
            if len(new) and float(new["ndx"].iloc[-1]) > 0:
                h = pd.concat([h[~h.index.isin(new.index)], new]).sort_index()
                note = name
                break
        except Exception:                                       # noqa: BLE001
            continue
    last_done = now.date() if (now.weekday() < 5 and now.hour * 60 + now.minute >= 16 * 60 + 20) else now.date() - dt.timedelta(days=1)
    h = h[h.index.date <= last_done]                            # today's row is still moving until the close
    h["vxn"] = h["vxn"].ffill(limit=5)
    h["qqq"] = h["qqq"].ffill()
    return h, note


def tbill_rate() -> float:
    try:
        raw = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS3MO")
        v = pd.to_numeric(raw.iloc[:, 1], errors="coerce").dropna()
        return float(v.iloc[-1]) / 100
    except Exception:                                           # noqa: BLE001
        return 0.04


# ---------------------------------------------------------------- demo market
def demo_expirations(now=None, weeks=9) -> list[dt.date]:
    """The coming Fridays (today's included until the close)."""
    now = now or now_ny()
    first = now.date() + dt.timedelta(days=(4 - now.weekday()) % 7)
    if first == now.date() and now.hour >= 16:
        first += dt.timedelta(days=7)
    return [first + dt.timedelta(days=7 * i) for i in range(weeks)]


def demo_chain(level: float, vxn: float, expiration: dt.date, now=None, r=0.04) -> pd.DataFrame:
    """A made-up XND chain built from a real index level and VXN: one-point strikes, volatility that rises for lower
    strikes the way the real chain's does, and bid-ask gaps about as wide as XND's."""
    from scipy.special import ndtr
    now = now or now_ny()
    T = max(((pd.Timestamp(expiration).tz_localize(NY) + pd.Timedelta(hours=16)) - now).total_seconds(), 1800.0) / (365 * 86400)
    # movement comes on trading days, so a Monday-to-Friday option holds a full week of it in under five calendar days
    today_left = 0.0
    if now.weekday() < 5:
        today_left = float(np.clip((16 * 60 - (now.hour * 60 + now.minute)) / 390.0, 0.0, 1.0))
    sessions = max(int(np.busday_count(now.date() + dt.timedelta(days=1), expiration + dt.timedelta(days=1))) + today_left, 0.05)
    root = math.sqrt(sessions / 252.0)
    base = vxn / 100.0
    F = level * math.exp(0.03 * T)
    wide = max(6.0 * base * root, 0.08)
    ks = np.arange(math.floor(F * (1 - wide)), math.ceil(F * (1 + 0.7 * wide)) + 1, 1.0)
    m = np.clip(np.log(ks / F) / (base * root), -4.0, 2.5)
    sd = base * np.maximum(0.93 - 0.16 * m + 0.04 * m * m, 0.75) * root
    d1 = (np.log(F / ks) + 0.5 * sd * sd) / sd
    put = math.exp(-r * T) * (ks * ndtr(-(d1 - sd)) - F * ndtr(-d1))
    call = put + math.exp(-r * T) * (F - ks)
    rows = []
    for right, price in (("PUT", put), ("CALL", call)):
        half = np.clip(0.10 + 0.02 * price, 0.03, 1.5)
        rows.append(pd.DataFrame({"strike": ks, "right": right, "bid": np.round(np.maximum(price - half, 0.0), 2),
                                  "ask": np.round(price + half, 2)}))
    q = pd.concat(rows, ignore_index=True)
    return q[q["ask"] >= 0.05].reset_index(drop=True)
