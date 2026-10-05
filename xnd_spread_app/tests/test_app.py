"""Runs the Streamlit app headlessly: first on demo quotes, then on a stand-in for ThetaData. Run with:  python tests/test_app.py"""
import datetime as dt
import os
import sys
import types

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import pandas as pd
from streamlit.testing.v1 import AppTest

import data

APP = os.path.join(HERE, "app.py")


def texts(at):
    out = []
    for kind in ("markdown", "caption", "success", "warning", "info", "error"):
        out += [e.value for e in getattr(at, kind)]
    return out


def metrics(at):
    return {m.label: (m.value, m.delta) for m in at.metric}


def widget(at, key):
    return next(w for kind in ("number_input", "selectbox") for w in getattr(at.sidebar, kind) if w.key == key)


# ---------------------------------------------------------------- demo quotes
at = AppTest.from_file(APP, default_timeout=180)
at.run()
assert not at.exception, at.exception
assert at.sidebar.toggle[0].value is True                      # no key -> demo quotes
m = metrics(at)
for label in ("XND (Nasdaq-100 ÷ 100)", "VXN, last close", "Index vs 200-day average", "Option quotes", "Options imply",
              "The model expects", "Options against the model", "You collect", "Most you can lose", "Chance you keep it all"):
    assert label in m, label
t = texts(at)
assert any("Demo quotes" in x for x in t)
assert sum("Stand aside this week" in x for x in t) == 2       # in the short answer and in the weekly detail
assert any("Weekly spreads lost money in every period tested" in x for x in t)
assert any(x.startswith("**Sell ") and "XND" in x and "put, buy" in x for x in t)
assert not any("$" in x.replace("\\$", "") for x in t), [x for x in t if "$" in x.replace("\\$", "")][:2]   # no bare dollar signs in running text
tables = [d.value for d in at.dataframe]
finder = [d for d in tables if "Chance (model)" in d.columns]
assert len(finder) == 2
for d, target, cap in ((finder[0], 125, 250_000 * 0.0075), (finder[1], 500, 250_000 * 0.03)):
    sell = d["Sell / buy"].str.split(" / ").str[0].astype(float)
    buy = d["Sell / buy"].str.split(" / ").str[1].astype(float)
    assert (d["Collect"] >= target).all() and (d["Max loss"] <= cap + 1e-6).all() and (buy < sell).all()
    assert sell.is_monotonic_increasing and d["Chance (options)"].between(50, 100).all()
print("CHECK demo: loads with no key and no errors; weekly says stand aside; both finders respect target and cap")
print("   ", {k: v for k, v in m.items() if k in ("Index vs 200-day average", "Options imply", "The model expects")})

# a target the cap can't reach says so, and says what the cap does allow
widget(at, "month_target").set_value(20_000)
at.run()
assert not at.exception, at.exception
w = [x.value for x in at.warning]
assert any("No spread collects" in x and "The most that cap allows" in x for x in w), w
print("CHECK an unreachable target:", next(x for x in w if "No spread collects" in x)[:150].replace("\\", ""))

# a cap smaller than one spread: no finder result and no tested trade for the month
widget(at, "month_target").set_value(500)
widget(at, "month_cap").set_value(0.1)
at.run()
assert not at.exception, at.exception
t = texts(at)
assert any("can't be built within your loss cap" in x for x in t)
widget(at, "month_cap").set_value(3.0)

# another expiry can be chosen; the month's caption follows it
box = widget(at, "month_exp")
choices = [e for e in data.demo_expirations(data.now_ny()) if 10 <= (e - data.now_ny().date()).days <= 60]
far = choices[-1]
box.set_value(far)
at.run()
assert not at.exception, at.exception
assert any(f"Expires {far:%a %b} {far.day}" in c.value for c in at.caption)
assert any("outside what was tested" in x for x in texts(at))
print(f"CHECK choosing {far} as the monthly expiry redraws everything and flags that it is outside what was tested")

# ---------------------------------------------------------------- stand-in for ThetaData
hist = data.saved_history()
LEVEL, VXN = float(hist["ndx"].iloc[-1]) / 100, float(hist["vxn"].iloc[-1])


class NoDataFoundError(Exception):
    pass


class FakeThetaClient:
    seen_keys, mode = [], "snapshot"

    def __init__(self, api_key=None, dataframe_type=None, **kw):
        FakeThetaClient.seen_keys.append(api_key)

    def option_list_expirations(self, symbol):
        assert symbol == "XND"
        today = data.now_ny().date()
        days = [today + dt.timedelta(days=i) for i in range(75)]
        return pd.DataFrame({"expiration": [str(d) for d in days if d.weekday() in (0, 2, 4)]})

    def _chain(self, expiration, when):
        q = data.demo_chain(LEVEL, VXN, pd.Timestamp(str(expiration)).date(), when)
        q["expiration"], q["timestamp"] = str(expiration), when
        return q

    def option_snapshot_quote(self, symbol, expiration):
        if FakeThetaClient.mode != "snapshot":
            raise NoDataFoundError("No data found")
        return self._chain(expiration, data.now_ny().floor("s"))

    def option_history_quote(self, symbol, expiration, interval, date):
        if FakeThetaClient.mode == "nothing":
            raise NoDataFoundError("No data found")
        return self._chain(expiration, pd.Timestamp(f"{date} 15:30", tz=data.NY))


fake = types.ModuleType("thetadata")
fake.ThetaClient = FakeThetaClient
sys.modules["thetadata"] = fake

live = AppTest.from_file(APP, default_timeout=180)
live.secrets["THETADATA_API_KEY"] = "test-key"
live.run()
assert not live.exception, live.exception
assert live.sidebar.toggle[0].value is False and FakeThetaClient.seen_keys == ["test-key"]
lm = metrics(live)
assert "live" in lm["Option quotes"][1] and not any("Demo quotes" in x for x in texts(live))
assert len([d for d in live.dataframe if "Chance (model)" in d.value.columns]) == 2
print("CHECK with a key in the secrets: quotes come from the client, marked live:", lm["Option quotes"])

FakeThetaClient.mode = "history"
live.sidebar.button[0].click()                                 # Refresh quotes
live.run()
assert not live.exception, live.exception
note = metrics(live)["Option quotes"][1]
assert "last session's close" in note or "latest on record" in note, note
print("CHECK no snapshot available: the app falls back and says so:", metrics(live)["Option quotes"])

FakeThetaClient.mode = "nothing"
live.sidebar.button[0].click()
live.run()
assert not live.exception, live.exception
assert sum("No usable XND quotes" in e.value for e in live.error) >= 2 and metrics(live)["Option quotes"][0] == "none"
print("CHECK no quotes at all: a plain message for each horizon, no crash")

# demo switched off without a key asks for the key and stops
nokey = AppTest.from_file(APP, default_timeout=60)
nokey.run()
nokey.sidebar.toggle[0].set_value(False)
nokey.run()
assert not nokey.exception and any("THETADATA_API_KEY" in e.value for e in nokey.sidebar.error)
print("CHECK demo off with no key: asks for the key in the app's secrets")
print("\nAll app checks passed.")
