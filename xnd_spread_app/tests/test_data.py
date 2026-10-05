"""Checks of the data layer with a stand-in for ThetaData (no key, no network). Run with:  python tests/test_data.py"""
import datetime as dt
import os
import re
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import numpy as np
import pandas as pd

import core
import data

NY = data.NY
EXP = dt.date(2026, 10, 30)
hist = data.saved_history()
LEVEL, VXN = float(hist["ndx"].iloc[-1]) / 100, float(hist["vxn"].iloc[-1])


class NoDataFoundError(Exception):
    pass


class FakeClient:
    """Answers like the ThetaData client: a snapshot (or not), and half-hour history for the days it is given."""

    def __init__(self, snapshot=True, history_days=(), snapshot_error=None, extra_expiration=False, naive_times=False, snapshot_time=None):
        self.snapshot, self.history_days, self.snapshot_error = snapshot, set(history_days), snapshot_error
        self.snapshot_time = snapshot_time or pd.Timestamp("2026-10-05 10:05:07", tz=NY)
        self.extra, self.naive, self.calls = extra_expiration, naive_times, []
        self.clock = pd.Timestamp("2100-01-01", tz=NY)          # set to "now" to hide half-hours that haven't happened yet

    def _chain(self, expiration, when):
        q = data.demo_chain(LEVEL, VXN, expiration, when)
        q["expiration"] = str(expiration)
        q["timestamp"] = when.tz_localize(None) if self.naive else when
        q["symbol"] = "XND"
        return q

    def option_list_expirations(self, symbol):
        self.calls.append(("list", symbol))
        days = [dt.date(2026, 9, 18)] + [dt.date(2026, 10, 5) + dt.timedelta(days=i) for i in range(60)]
        return pd.DataFrame({"symbol": symbol, "expiration": [str(d) for d in days if d.weekday() in (0, 2, 4)]})

    def option_snapshot_quote(self, symbol, expiration):
        self.calls.append(("snapshot", expiration))
        if self.snapshot_error:
            raise self.snapshot_error
        if not self.snapshot:
            raise NoDataFoundError("No data found for: option_snapshot_quote")
        q = self._chain(expiration, self.snapshot_time)
        if self.extra:
            q = pd.concat([q, self._chain(dt.date(2026, 11, 6), self.snapshot_time)])
        return q

    def option_history_quote(self, symbol, expiration, interval, date):
        self.calls.append(("history", date))
        if date not in self.history_days:
            raise NoDataFoundError("No data found for: option_history_quote")
        bars = [pd.Timestamp(f"{date} {t}", tz=NY) for t in ("09:30", "12:00", "15:30") if pd.Timestamp(f"{date} {t}", tz=NY) <= self.clock]
        return pd.concat([self._chain(expiration, b) for b in bars], ignore_index=True)


now = pd.Timestamp("2026-10-05 10:06", tz=NY)                 # a Monday morning

# ---- expirations
exps = data.list_expirations(FakeClient(), today=now.date())
assert exps[0] == dt.date(2026, 10, 5) and dt.date(2026, 9, 18) not in exps and exps == sorted(exps)
ends = core.week_ends(exps, now.date())
assert ends[0] == dt.date(2026, 10, 9) and ends[core.pick_monthly(ends, now.date())] == EXP
print("CHECK expirations: past dates dropped; this week's is Fri Oct 9, the month's is Fri Oct 30")

# ---- quotes: live snapshot
c = FakeClient()
q, source, ts, notes = data.fetch_chain(c, EXP, now=now)
assert source == "live" and ts == pd.Timestamp("2026-10-05 10:05:07", tz=NY) and len(q) > 100
assert [x[0] for x in c.calls] == ["snapshot"]
ch = core.clean_chain(q, near=LEVEL)
T = core.years_to(EXP, ts)
F = core.forward_level(ch, 0.04, T)
assert abs(F / LEVEL - 1) < 0.01
print(f"CHECK live snapshot: {len(q)} quotes at {ts:%H:%M:%S}, index level from the options {F:.2f} (index {LEVEL:.2f})")

# a response that mixes in another expiration only keeps the one asked for
q2, _, _, notes2 = data.fetch_chain(FakeClient(extra_expiration=True), EXP, now=now)
assert len(q2) == len(q) and notes2["expirations returned"] == 2 and notes2["rows used"] == len(q)
print("CHECK a response holding two expirations keeps only the one asked for")

# ---- quotes: no snapshot (plan or hour) -> today's latest half-hour bar
c = FakeClient(snapshot=False, history_days={now.date()})
c.clock = pd.Timestamp("2026-10-05 13:10", tz=NY)
q, source, ts, _ = data.fetch_chain(c, EXP, now=c.clock)
assert source == "earlier today" and ts == pd.Timestamp("2026-10-05 12:00", tz=NY)
assert not q.duplicated(["strike", "right"]).any() and (q["timestamp"] == ts).all()
# a snapshot that errors for any other reason (a plan without snapshots) falls back the same way and is noted
c = FakeClient(snapshot_error=PermissionError("snapshots need a higher plan"), history_days={now.date()})
c.clock = now
q, source, ts, notes = data.fetch_chain(c, EXP, now=now)
assert source == "earlier today" and "PermissionError" in notes["snapshot"]
print("CHECK no snapshot: falls back to today's latest quotes on record and says why")

# ---- quotes: weekend -> the last session's close
sunday = pd.Timestamp("2026-10-04 20:00", tz=NY)
c = FakeClient(snapshot=False, history_days={dt.date(2026, 10, 2)})
q, source, ts, _ = data.fetch_chain(c, EXP, now=sunday)
assert source == "last close" and ts == pd.Timestamp("2026-10-02 15:30", tz=NY)
assert [x for x in c.calls if x[0] == "history"][0][1] == dt.date(2026, 10, 2)        # Saturday and Sunday are never asked for
assert core.sessions_to(EXP, ts) == 21 and core.sessions_to(EXP, pd.Timestamp("2026-10-02 16:00", tz=NY)) == 20
# a snapshot taken after the close (the day's last, widest quotes) gives way to the session's final full half-hour
friday_late = pd.Timestamp("2026-10-02 16:14:30", tz=NY)
c = FakeClient(snapshot_time=friday_late, history_days={dt.date(2026, 10, 2)})
q, source, ts, _ = data.fetch_chain(c, EXP, now=sunday)
assert source == "last close" and ts == pd.Timestamp("2026-10-02 15:30", tz=NY)
# ...unless there is no history to use, in which case the snapshot is kept and labelled
q, source, ts, _ = data.fetch_chain(FakeClient(snapshot_time=friday_late), EXP, now=sunday)
assert source == "after hours" and ts == friday_late and len(q) > 100
# ...or the only history on record is from an earlier session than the snapshot
q, source, ts, _ = data.fetch_chain(FakeClient(snapshot_time=friday_late, history_days={dt.date(2026, 10, 1)}), EXP, now=sunday)
assert source == "after hours" and ts == friday_late
# the same evening, after the close
evening = pd.Timestamp("2026-10-02 18:00", tz=NY)
q, source, ts, _ = data.fetch_chain(FakeClient(snapshot_time=friday_late, history_days={dt.date(2026, 10, 2)}), EXP, now=evening)
assert source == "last close" and ts == pd.Timestamp("2026-10-02 15:30", tz=NY)
assert data.market_open(now) and not data.market_open(sunday) and not data.market_open(evening)
assert data.market_open(pd.Timestamp("2026-10-05 16:10", tz=NY)) and not data.market_open(pd.Timestamp("2026-10-05 09:20", tz=NY))
print("CHECK market closed: a snapshot from 4:14 pm gives way to the session's last full half-hour; kept and labelled if that is all there is")
# plain New York clock times (no time zone) are read as New York time
q, source, ts, _ = data.fetch_chain(FakeClient(snapshot=False, history_days={dt.date(2026, 10, 2)}, naive_times=True), EXP, now=sunday)
assert ts == pd.Timestamp("2026-10-02 15:30", tz=NY)
# before the open on a Monday the last session is Friday
assert data.recent_sessions(pd.Timestamp("2026-10-05 08:00", tz=NY))[0] == dt.date(2026, 10, 2)
assert data.recent_sessions(now)[:2] == [dt.date(2026, 10, 5), dt.date(2026, 10, 2)]
# nothing anywhere
q, source, ts, _ = data.fetch_chain(FakeClient(snapshot=False), EXP, now=now)
assert source == "none" and ts is None and not len(q)
print("CHECK weekend and pre-market use Friday's close; times without a zone are read as New York; no data is reported as none")

# ---- index history
def fails(start):
    raise ConnectionError("no network")


h, src = data.load_history(now=now, sources=(("Yahoo Finance", fails), ("FRED", fails)))
assert src == "saved copy only" and h.index[-1] == hist.index[-1] and len(h) == len(hist)


def fresh(start):
    days = pd.to_datetime(["2026-10-01", "2026-10-02", "2026-10-05", "2026-10-06"])
    return pd.DataFrame({"ndx": [30501.56, 30807.93, 31000.0, 31100.0], "vxn": [22.51, 21.2, 20.5, np.nan], "qqq": [742.03, 749.58, 754.2, 756.6]}, index=days)


tuesday_noon = pd.Timestamp("2026-10-06 12:00", tz=NY)
h, src = data.load_history(now=tuesday_noon, sources=(("Yahoo Finance", fails), ("FRED", fresh)))
assert src == "FRED" and h.index[-1] == pd.Timestamp("2026-10-05") and h["ndx"].iloc[-1] == 31000.0      # today's row is still moving
assert not h.index.duplicated().any() and len(h) == len(hist) + 1
after_close = pd.Timestamp("2026-10-06 16:30", tz=NY)
h, _ = data.load_history(now=after_close, sources=(("Yahoo Finance", fresh),))
assert h.index[-1] == pd.Timestamp("2026-10-06") and h["vxn"].iloc[-1] == 20.5                            # a missing VXN print is carried forward
print("CHECK history: saved copy when nothing answers; new days are added; today's unfinished day is left out until the close")

# ---- demo chain
dq = data.demo_chain(LEVEL, VXN, dt.date(2026, 10, 9), now)
dc = core.clean_chain(dq, near=LEVEL)
Tw = core.years_to(dt.date(2026, 10, 9), now)
Fw = core.forward_level(dc, 0.04, Tw)
puts = core.puts_table(dc, Fw, Tw, 0.04)
assert abs(Fw / LEVEL - 1) < 0.01 and (dq["ask"] >= dq["bid"]).all() and puts["delta"].between(-1, 0).all()
move = core.atm_vol(dc, Fw, Tw, 0.04) * Tw ** 0.5
assert 0.6 * VXN / 100 * (5 / 252) ** 0.5 < move < 1.2 * VXN / 100 * (5 / 252) ** 0.5
print(f"CHECK demo chain: {len(dq)} quotes, a week's implied move of ±{move:.1%} with VXN at {VXN}")

# ---- the key is nowhere in the files
found = []
for root, _, files in os.walk(HERE):
    if "__pycache__" in root:
        continue
    for name in files:
        if name.endswith((".csv", ".png", ".pyc")):
            continue
        text = open(os.path.join(root, name), encoding="utf-8", errors="ignore").read()
        for m in re.finditer(r"THETADATA_API_KEY\s*=\s*[\"']([^\"']+)[\"']", text):
            if m.group(1) not in ("your-thetadata-api-key", "test-key", "..."):
                found.append((name, m.group(1)[:6]))
        if re.search(r"api_key\s*=\s*[\"'][A-Za-z0-9_\-]{16,}[\"']", text):
            found.append((name, "api_key literal"))
assert not found, found
assert ".streamlit/secrets.toml" in open(os.path.join(HERE, ".gitignore")).read()
print("CHECK no API key in any file; secrets.toml is ignored by git")
print("\nAll data checks passed.")
