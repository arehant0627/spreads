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
CLOCK = [pd.Timestamp("2026-10-05 10:30", tz=data.NY)]         # a Monday morning with the market open, so results don't depend on today
data.now_ny = lambda: CLOCK[0]


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
for label in ("XND level", "VXN", "vs 200-day average", "Options imply", "Model expects", "Options vs model"):
    assert label in m, label
    assert len(label) <= 18 and len(m[label][0]) <= 8 and len(m[label][1] or "") <= 16, (label, m[label])   # short enough for a narrow window
t = texts(at)
assert any("Demo quotes" in x for x in t)
assert not any("tand aside this week" in x for x in t)          # the week gets the model's view, not a blanket stand-aside
assert sum(x.startswith("This week's spread:") for x in t) == 2   # in the short answer and in the weekly detail
assert not any("backtest" in x.lower() for x in t) and not any("Behind the numbers" in h.value for h in at.subheader)
assert not any(lab in e.label for e in at.expander for lab in ("How good is the model?", "What the backtests found", "Data check"))
assert any(x == "#### Recommended trade" for x in t)
assert sum("How it is chosen: of the" in c.value and "allowed pairs of puts" in c.value for c in at.caption) == 2
assert sum("For comparison, the fixed 20/10-delta pair" in c.value for c in at.caption) == 2
assert not any("closest to 20 delta" in c.value for c in at.caption)
assert any(x.startswith("**Sell ") and "XND" in x and "put, buy" in x for x in t)
assert sum(x.startswith("**Your ") and "target:**" in x for x in t) == 2 and sum(x.startswith("**Recommended trade:**") for x in t) == 2
assert sum("Possible inside your cap" in x for x in t) == 2 and any("- **Most you can lose:**" in x for x in t)
assert any("Option quotes:" in c.value and "made-up demo quotes" in c.value for c in at.caption)
assert not any("$" in x.replace("\\$", "") for x in t), [x for x in t if "$" in x.replace("\\$", "")][:2]   # no bare dollar signs in running text
tables = [d.value for d in at.dataframe]
finder = [d for d in tables if "Chance (model)" in d.columns]
assert len(finder) == 2
for d, target, cap in ((finder[0], 125, 250_000 * 0.0075), (finder[1], 500, 250_000 * 0.03)):
    sell = d["Sell / buy"].str.split(" / ").str[0].astype(float)
    buy = d["Sell / buy"].str.split(" / ").str[1].astype(float)
    assert (d["Collect"] >= target).all() and (d["Max loss"] <= cap + 1e-6).all() and (buy < sell).all()
    assert sell.is_monotonic_increasing and d["Chance (options)"].between(50, 100).all()
print("CHECK demo: loads with no key and no errors; no stand-aside on the week, no backtest or background sections; both finders respect target and cap")
print("    week:", next(x for x in t if x.startswith("This week's spread:")))
print("   ", {k: v for k, v in m.items() if k in ("vs 200-day average", "Options imply", "Model expects")})

# a target the cap can't reach: says not possible, and shows the risk level and the spread that would reach it
widget(at, "week_target").set_value(1000)
at.run()
assert not at.exception, at.exception
errs = [e.value for e in at.error]
msg = next(x for x in errs if "Not possible inside your cap" in x)
assert "takes at least" in msg and "of the account at risk" in msg and "the most it can collect is about" in msg, msg
t = texts(at)
verdict = next(x for x in t if x.startswith("**Your ") and "1,000 target:**" in x)
assert "not possible inside your cap" in verdict and "at risk" in verdict, verdict
assert any(x.startswith("**How to get there at the lowest risk level: Sell ") for x in t)
ways = next(d.value for d in at.dataframe if "% at risk" in d.value.columns)
cap_week = 250_000 * 0.0075
assert (ways["Collect"] >= 1000).all() and (ways["Max loss"] > cap_week).all() and ways["Max loss"].is_monotonic_increasing
assert abs(ways["% at risk"].iloc[0] - ways["Max loss"].iloc[0] / 2500) < 1e-9
need_pct = float(ways["% at risk"].iloc[0])
print("CHECK an unreachable target:", msg[:175].replace("\\", "").replace("**", ""))

# raise the cap to the level it named and the same target becomes possible
widget(at, "week_cap").set_value(round(need_pct + 0.06, 2))
at.run()
assert not at.exception, at.exception
t = texts(at)
verdict = next(x for x in t if x.startswith("**Your ") and "1,000 target:**" in x)
assert "not possible" not in verdict and "chance of keeping it all" in verdict, verdict
print(f"CHECK raising the weekly cap to {need_pct + 0.06:.2f}% (the level it named) makes the $1,000 target possible")

# a target no quotes can reach at any risk level
widget(at, "month_target").set_value(50_000_000)
at.run()
assert not at.exception, at.exception
assert any("Not possible with these quotes at any risk level" in e.value for e in at.error)
widget(at, "week_target").set_value(125)
widget(at, "week_cap").set_value(0.75)

# a cap smaller than one spread: no tested trade for the month, and the reason is the cap
widget(at, "month_target").set_value(500)
widget(at, "month_cap").set_value(0.1)
at.run()
assert not at.exception, at.exception
t = texts(at)
assert any("one spread risks more than your loss cap" in x for x in t)
assert any("Not possible inside your cap" in e.value and "Your cap is 0.1% of the account" in e.value for e in at.error)
# a cap above 20% of the account is accepted (it used to be refused), and a big target then fits
widget(at, "month_cap").set_value(35.0)
widget(at, "month_target").set_value(5000)
at.run()
assert not at.exception, at.exception
big = next(d.value for d in at.dataframe if "Chance (model)" in d.value.columns and (d.value["Collect"] >= 5000).all())
assert (big["Max loss"] <= 250_000 * 0.35 + 1e-6).all() and big["Max loss"].max() > 250_000 * 0.20
assert any("Possible inside your cap of 35% of the account" in s.value for s in at.success)
print(f"CHECK a 35% loss cap is accepted: $5,000 a month fits, with up to ${big['Max loss'].max():,.0f} at risk")
widget(at, "month_cap").set_value(3.0)
widget(at, "month_target").set_value(500)
print("CHECK a target beyond any risk level, and a cap smaller than one spread, each get a plain message")

# another expiry can be chosen; the month's caption follows it
box = widget(at, "month_exp")
choices = [e for e in data.demo_expirations(data.now_ny()) if 10 <= (e - data.now_ny().date()).days <= 60]
far = choices[-1]
box.set_value(far)
at.run()
assert not at.exception, at.exception
assert any(f"Expires {far:%a %b} {far.day}" in c.value for c in at.caption)
assert any("normally opened about 25 days out" in x for x in texts(at))
print(f"CHECK choosing {far} as the monthly expiry redraws everything and notes it is further out than usual")

# ---------------------------------------------------------------- stand-in for ThetaData
hist = data.saved_history()
LEVEL, VXN = float(hist["ndx"].iloc[-1]) / 100, float(hist["vxn"].iloc[-1])


class NoDataFoundError(Exception):
    pass


class FakeThetaClient:
    seen_keys, mode, snapshot_time = [], "snapshot", None

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
        return self._chain(expiration, FakeThetaClient.snapshot_time or data.now_ny().floor("s"))

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
quote_line = lambda a: next(c.value for c in a.caption if c.value.startswith("Option quotes:"))
assert quote_line(live).endswith("live.") and not any("Demo quotes" in x for x in texts(live))
assert len([d for d in live.dataframe if "Chance (model)" in d.value.columns]) == 2
print("CHECK with a key in the secrets: quotes come from the client, marked live:", quote_line(live))

FakeThetaClient.mode = "history"
live.sidebar.button[0].click()                                 # Refresh quotes
live.run()
assert not live.exception, live.exception
note = quote_line(live)
assert "latest on record today" in note, note
print("CHECK no snapshot available: the app falls back and says so:", note)

# Sunday night: the snapshot is Friday's 4:14 pm leftovers, so the session's last full half-hour is used and the app says the market is closed
CLOCK[0] = pd.Timestamp("2026-10-04 21:00", tz=data.NY)
FakeThetaClient.mode, FakeThetaClient.snapshot_time = "snapshot", pd.Timestamp("2026-10-02 16:14", tz=data.NY)
live.sidebar.button[0].click()
live.run()
assert not live.exception, live.exception
note = quote_line(live)
assert "Fri Oct 2, 3:30 PM ET" in note and "final half-hour" in note, note
assert any("The market is closed" in i.value for i in live.info)
print("CHECK market closed:", note, "| the app says to check again after the open")
CLOCK[0] = pd.Timestamp("2026-10-05 10:30", tz=data.NY)
FakeThetaClient.snapshot_time = None

FakeThetaClient.mode = "nothing"
live.sidebar.button[0].click()
live.run()
assert not live.exception, live.exception
assert sum("No usable XND quotes" in e.value for e in live.error) >= 2 and quote_line(live) == "Option quotes: none."
print("CHECK no quotes at all: a plain message for each horizon, no crash")

# demo switched off without a key asks for the key and stops
nokey = AppTest.from_file(APP, default_timeout=60)
nokey.run()
nokey.sidebar.toggle[0].set_value(False)
nokey.run()
assert not nokey.exception and any("THETADATA_API_KEY" in e.value for e in nokey.sidebar.error)
print("CHECK demo off with no key: asks for the key in the app's secrets")
print("\nAll app checks passed.")
