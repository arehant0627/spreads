"""Checks of the planner's logic. Run with:  python tests/test_core.py"""
import datetime as dt
import math
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import numpy as np
import pandas as pd
from scipy.special import ndtr

import core
import data

NY = "America/New_York"

# ---- option maths
F, T, r = 300.0, 25 / 365, 0.04
K = np.array([270.0, 285.0, 300.0, 315.0])
vols = np.array([0.32, 0.26, 0.20, 0.17])
back = core.put_iv(core.black76_put(F, K, T, r, vols), F, K, T, r)
assert np.allclose(back, vols, atol=1e-6), back
assert np.isnan(core.put_iv([0.0], F, [250.0], T, r)[0])           # a price no volatility can produce
print("CHECK implied volatility recovers the volatility a price was made with")

# ---- a chain made from a known distribution (two bell curves mixed: a calm one and a rough one that sits lower)
w = np.array([0.8, 0.2]); f = np.array([304.0, 284.0]); v = np.array([0.14, 0.34])
assert abs((w * f).sum() - F) < 1e-9
ks = np.arange(230.0, 361.0, 1.0)


def true_put(k):
    return sum(wi * core.black76_put(fi, k, T, r, vi) for wi, fi, vi in zip(w, f, v))


def true_below(k):
    return sum(wi * ndtr(-(np.log(fi / k) - 0.5 * vi * vi * T) / (vi * math.sqrt(T))) for wi, fi, vi in zip(w, f, v))


put = true_put(ks)
call = put + math.exp(-r * T) * (F - ks)
raw = pd.concat([pd.DataFrame({"strike": ks, "right": "PUT", "bid": np.maximum(put - 0.03, 0).round(2), "ask": (put + 0.03).round(2)}),
                 pd.DataFrame({"strike": ks, "right": "CALL", "bid": np.maximum(call - 0.03, 0).round(2), "ask": (call + 0.03).round(2)})])
chain = core.clean_chain(raw, near=300)
assert len(chain) == 2 * len(ks) and set(chain["side"]) == {"P", "C"}
got_F = core.forward_level(chain, r, T)
assert abs(got_F - F) < 0.05, got_F
smile = core.smile_table(chain, got_F, T, r)
mid = smile[(smile["p_below"] > 0.04) & (smile["p_below"] < 0.96)]
err = np.abs(mid["p_below"].to_numpy() - true_below(mid["strike"].to_numpy()))
assert err.max() < 0.02, err.max()
assert (np.diff(smile["p_below"]) >= -1e-12).all()
plain = core.puts_table(chain, got_F, T, r, smile=pd.DataFrame())           # no allowance for the smile
with_smile = core.puts_table(chain, got_F, T, r, smile)
k20 = with_smile.iloc[(with_smile["delta"] + 0.20).abs().argmin()]
p20 = plain[plain["strike"] == k20["strike"]].iloc[0]
truth = 1 - true_below(k20["strike"])
assert abs(k20["p_above"] - truth) < 0.02 < abs(p20["p_above"] - truth), (k20["p_above"], p20["p_above"], truth)
print(f"CHECK chance of finishing above the 20-delta strike: truth {truth:.3f}, with the smile {k20['p_above']:.3f}, without {p20['p_above']:.3f}")
imp = core.implied_range(smile, got_F, T, core.atm_vol(chain, got_F, T, r))
grid = np.arange(230.0, 360.0, 0.01)
for q in (0.10, 0.25, 0.75, 0.90):
    true_q = grid[np.searchsorted(true_below(grid), q)] / F - 1
    assert abs(imp[f"q{int(q * 100)}"] - true_q) < 0.006, (q, imp[f"q{int(q * 100)}"], true_q)
assert imp["from_smile"] and imp["q10"] < imp["q25"] < imp["q50"] < imp["q75"] < imp["q90"]
short_chain = chain[(chain["strike"] > 290) & (chain["strike"] < 310)]
fallback = core.implied_range(core.smile_table(short_chain, got_F, T, r), got_F, T, 0.2)
assert not fallback["from_smile"] and fallback["q10"] < 0 < fallback["q90"]
print("CHECK the options' range matches the distribution the prices were made from; falls back when quotes stop short")

# strikes in thousandths are spotted
thousandths = raw.assign(strike=raw["strike"] * 1000)
assert core.clean_chain(thousandths, near=300)["strike"].max() == ks.max()
# crossed or empty quotes are dropped
bad = pd.DataFrame({"strike": [300, 301, 302], "right": ["PUT"] * 3, "bid": [2.0, 3.0, None], "ask": [1.5, 0.0, 2.0]})
assert core.clean_chain(bad)["strike"].tolist() == [302]
print("CHECK cleaning: thousandths, crossed and empty quotes")

# ---- spreads
puts = with_smile
one = core.spread_numbers(puts[puts["strike"] == 290].iloc[0], puts[puts["strike"] == 280].iloc[0], "quote", 1.0)
s290, s280 = puts[puts["strike"] == 290].iloc[0], puts[puts["strike"] == 280].iloc[0]
assert abs(one["credit"] - (s290["bid"] - s280["ask"])) < 1e-12 and one["width"] == 10
assert abs(one["net"] - (one["credit"] * 100 - 2)) < 1e-9 and abs(one["risk"] - ((10 - one["credit"]) * 100 + 2)) < 1e-9
assert abs(one["net"] + one["risk"] - 1000) < 1e-9                  # what you collect plus what you can lose is the width
assert core.spread_numbers(s290, s280, "mid")["credit"] > one["credit"]

target, cap = 500.0, 7500.0
table, info = core.find_spreads(puts, target, cap)
assert len(table) > 3
assert (table["premium"] >= target - 1e-9).all() and (table["max_loss"] <= cap + 1e-9).all()
assert (table["long"] < table["short"]).all() and table["short"].is_monotonic_increasing and (table["delta"].abs() <= 0.30).all()
assert table["p_above"].is_monotonic_decreasing                       # safest first
for _, row in table.iterrows():                                        # the bought strike is the one that risks the least
    s = puts[puts["strike"] == row["short"]].iloc[0]
    best = None
    for _, l in puts[puts["strike"] < row["short"]].iterrows():
        x = core.spread_numbers(s, l)
        if x["net"] <= 0:
            continue
        n = math.ceil(target / x["net"])
        if n * x["risk"] <= cap and (best is None or n * x["risk"] < best):
            best = n * x["risk"]
    assert abs(best - row["max_loss"]) < 1e-6, (row["short"], best, row["max_loss"])
    assert abs(row["premium"] - row["n"] * (row["credit"] * 100 - 2)) < 1e-6
none, info2 = core.find_spreads(puts, 50_000.0, cap)
assert not len(none) and info2["most_within_cap"]["premium"] < 50_000 and info2["most_within_cap"]["n"] >= 1
tiny, info3 = core.find_spreads(puts, 100.0, 50.0)                     # a cap smaller than any one spread
assert not len(tiny) and info3["most_within_cap"] is None
print(f"CHECK finder: {len(table)} spreads reach ${target:,.0f} inside ${cap:,.0f}, each with the least-risk bought strike; "
      f"an impossible target reports the most the cap allows (${info2['most_within_cap']['premium']:,.0f})")

t = core.tested_spread(puts, cap)
assert abs(abs(t["delta"]) - 0.20) < 0.02 and abs(abs(t["long_delta"]) - 0.10) < 0.02
assert t["n"] == int(cap // t["risk_each"]) and t["max_loss"] <= cap
assert core.tested_spread(puts, 10.0)["n"] == 0
assert core.tested_spread(puts.iloc[:0], cap) is None
print(f"CHECK tested spread: sell {t['short']:g} ({t['delta']:.2f}), buy {t['long']:g} ({t['long_delta']:.2f}), {t['n']} spreads")

# ---- the range model on the real history saved with the app
hist = data.saved_history()
assert len(hist) > 6000 and hist.index.is_monotonic_increasing and not hist["ndx"].isna().any()
for h in (1, 4, 5, 18, 20, 25):
    m = core.fit_range_model(hist["ndx"], hist["vxn"], h)
    k = m["check"]
    assert 0.74 <= k["inside"] <= 0.84, (h, k["inside"])
    assert 0.07 <= k["below"] <= 0.13, (h, k["below"])
    assert k["inside_vxn"] > k["inside"] and k["error_model"] < k["error_vxn"]
    assert k["size_low"] < 1 < k["size_high"] and 0.05 < m["sigma"] < 1.5
    rng = core.model_range(m)
    assert rng["q10"] < rng["q25"] < rng["q50"] < rng["q75"] < rng["q90"] and rng["move_low"] < rng["move"] < rng["move_high"]
    print(f"CHECK model, {h:>2} sessions: its 8-in-10 range held {k['inside']:.0%} of the time ({k['below']:.0%} below); "
          f"VXN alone {k['inside_vxn']:.0%}; size error {k['error_model']:.2f} vs {k['error_vxn']:.2f}")

# no peeking: a fit that never saw 2016 onward gives the same out-of-sample results for earlier years
full = core.fit_range_model(hist["ndx"], hist["vxn"], 20)
cut = hist[hist.index <= "2015-12-31"]
early = core.fit_range_model(cut["ndx"], cut["vxn"], 20)
a, b = full["oos"][full["oos"].index.year <= 2014], early["oos"][early["oos"].index.year <= 2014]
assert len(a) == len(b) > 2000 and np.allclose(a.to_numpy(), b.to_numpy())
print("CHECK no peeking: results for 2006-14 are identical whether or not the fit is given 2016-26")

level = 300.0
ca = [core.chance_above(full, level, k) for k in (270, 280, 290, 300)]
assert all(x["lo"] <= x["p"] <= x["hi"] for x in ca) and ca[0]["p"] > ca[1]["p"] > ca[2]["p"] > ca[3]["p"]
lo, hi = core.wilson(0.9, 250)
assert 0.87 < lo < 0.9 < hi < 0.93 and core.wilson(0.5, 0) == (0.0, 1.0)
sp = dict(short=285.0, long=275.0, credit=0.80, width=10.0)
val = core.model_value(full, level, sp, 1.0)
assert val["lo"] < val["mean"] < val["hi"] and 0 <= val["full_loss"] <= 1 - val["win"] <= 1
breakeven = 285 - (0.80 * 100 - 2) / 100
assert abs(val["win"] - core.chance_above(full, level, breakeven)["p"]) < 0.002
assert abs(val["full_loss"] - (1 - core.chance_above(full, level, 275.0)["p"])) < 0.002
free = core.model_value(full, level, dict(short=285.0, long=275.0, credit=9.9, width=10.0), 0.0)
assert free["mean"] > val["mean"]                                       # more premium for the same strikes is worth more
print(f"CHECK model's value: mean {val['mean']:+.3f} per $1 at risk, pays {val['win']:.0%}, full loss {val['full_loss']:.1%}")

# ---- the rule and the recommendation
tr = core.trend_state(hist["ndx"])
assert abs(tr["average"] - hist["ndx"].iloc[-200:].mean()) < 1e-9 and tr["above"] == (tr["last"] >= tr["average"])
up, down = dict(tr, above=True, gap=0.05), dict(tr, above=False, gap=-0.05)
good = dict(n=5)
assert core.recommend("week", up, good, val)["action"] == "stand aside"
assert core.recommend("month", down, good, val)["action"] == "stand aside"
rec = core.recommend("month", up, good, val, days_left=25)
assert rec["action"] == "trade" and not any("outside what was tested" in n for n in rec["notes"])
assert any("outside what was tested" in n for n in core.recommend("month", up, good, val, days_left=46)["notes"])
assert core.recommend("month", up, dict(n=0), val)["action"] == "stand aside"
assert core.recommend("month", up, None, None)["action"] == "stand aside"
assert "thin" in core.recommend("month", up, good, dict(val, hi=-0.01))["notes"][1]
print("CHECK recommendation: weekly stands aside; monthly follows the 200-day rule; no trade when it can't be built")

# ---- calendar
assert core.third_friday(2026, 10) == dt.date(2026, 10, 16) and core.third_friday(2026, 11) == dt.date(2026, 11, 20)
assert core.is_monthly(dt.date(2026, 10, 16)) and not core.is_monthly(dt.date(2026, 10, 9))
assert core.is_monthly(dt.date(2025, 4, 17))                            # Good Friday was the third Friday in April 2025
listed = [dt.date(2026, 10, 5) + dt.timedelta(days=i) for i in range(70) if (dt.date(2026, 10, 5) + dt.timedelta(days=i)).weekday() in (0, 2, 4)]
ends = core.week_ends(listed, dt.date(2026, 10, 5))
assert ends[:4] == [dt.date(2026, 10, 9), dt.date(2026, 10, 16), dt.date(2026, 10, 23), dt.date(2026, 10, 30)]
monday, friday = pd.Timestamp("2026-10-05 10:00", tz=NY), pd.Timestamp("2026-10-09 11:00", tz=NY)
assert core.pick_weekly(ends, monday) == 0
assert core.pick_weekly(core.week_ends(listed, friday.date()), friday) == 1
assert ends[core.pick_monthly(ends, monday.date())] == dt.date(2026, 10, 30)
assert core.sessions_to(dt.date(2026, 10, 9), monday) == 5
assert core.sessions_to(dt.date(2026, 10, 9), pd.Timestamp("2026-10-05 16:30", tz=NY)) == 4
assert core.sessions_to(dt.date(2026, 10, 9), pd.Timestamp("2026-10-03 12:00", tz=NY)) == 5
assert abs(core.years_to(dt.date(2026, 10, 9), monday) - 4.25 / 365) < 1e-9
print("CHECK calendar: third Fridays, week-ending expiries, which week and month are picked, sessions left")
print("\nAll core checks passed.")
