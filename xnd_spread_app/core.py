"""Logic for the XND put spread planner. No network and no Streamlit in here, so every function can be tested on its own.

Three parts:
  1. option maths and cleaning one expiration's quotes,
  2. the spread finder (best chance to collect a target premium within a loss cap) and the trade the backtests used,
  3. the range model: how far the index is likely to move by expiry, from 25 years of history.
"""
from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pandas as pd
from scipy.special import ndtr

MULTIPLIER = 100               # dollars per index point, per XND contract
Z80 = 1.2815515655446004       # +/- this many standard deviations holds 80% of a normal distribution
TESTED_SHORT_DELTA, TESTED_LONG_DELTA = 0.20, 0.10      # the spread every backtest used
TREND_DAYS = 200               # the rule that passed the 2020-26 tests: trade only above this average


# ---------------------------------------------------------------- option maths
def black76_put(F, K, T, r, sig):
    F, K, sig = np.asarray(F, float), np.asarray(K, float), np.asarray(sig, float)
    if T <= 0:
        return np.maximum(K - F, 0.0)
    sd = np.maximum(sig, 1e-9) * math.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sd * sd) / sd
    return math.exp(-r * T) * (K * ndtr(-(d1 - sd)) - F * ndtr(-d1))


def put_iv(price, F, K, T, r):
    """Implied volatility by bisection; NaN where no volatility can produce the price."""
    price, K = np.atleast_1d(np.asarray(price, float)), np.atleast_1d(np.asarray(K, float))
    lo, hi = np.full(len(K), 0.01), np.full(len(K), 5.0)
    ok = (price > black76_put(F, K, T, r, lo) + 1e-12) & (price < black76_put(F, K, T, r, hi))
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        above = black76_put(F, K, T, r, mid) > price
        hi = np.where(above, mid, hi)
        lo = np.where(above, lo, mid)
    return np.where(ok, 0.5 * (lo + hi), np.nan)


def years_to(expiration: dt.date, now: pd.Timestamp) -> float:
    """Calendar time from now to 4pm New York on the expiration day, in years (never less than half an hour)."""
    end = pd.Timestamp(expiration).tz_localize(now.tz) + pd.Timedelta(hours=16)
    return max((end - now).total_seconds(), 1800.0) / (365.0 * 86400.0)


def sessions_to(expiration: dt.date, now: pd.Timestamp) -> int:
    """Trading sessions left until expiry, counting today if the market has not closed (weekends skipped, holidays not)."""
    today = now.date()
    start = today if (now.weekday() < 5 and now.hour < 16) else today + dt.timedelta(days=1)
    return max(int(np.busday_count(start, expiration + dt.timedelta(days=1))), 1)


# ---------------------------------------------------------------- one expiration's quotes
def clean_chain(q: pd.DataFrame, near: float | None = None) -> pd.DataFrame:
    """Raw quotes (strike, right, bid, ask) -> one row per strike and side with numeric columns and a mid.
    `near` is roughly where the index is; it is only used to spot strikes sent in thousandths of a dollar."""
    if q is None or not len(q):
        return pd.DataFrame(columns=["strike", "side", "bid", "ask", "mid"])
    d = pd.DataFrame({"strike": pd.to_numeric(q["strike"], errors="coerce"),
                      "side": q["right"].astype(str).str.upper().str[0],
                      "bid": pd.to_numeric(q["bid"], errors="coerce"), "ask": pd.to_numeric(q["ask"], errors="coerce")})
    d = d.dropna(subset=["strike", "ask"])
    d["bid"] = d["bid"].fillna(0.0)
    if len(d) and near and d["strike"].median() > 100 * near:   # strikes sent in thousandths of a dollar
        d["strike"] = d["strike"] / 1000.0
    d = d[(d["ask"] > 0) & (d["bid"] >= 0) & (d["ask"] >= d["bid"]) & d["side"].isin(["P", "C"])]
    d = d.drop_duplicates(["strike", "side"]).sort_values(["side", "strike"]).reset_index(drop=True)
    d["mid"] = 0.5 * (d["bid"] + d["ask"])
    return d


def forward_level(chain: pd.DataFrame, r: float, T: float) -> float:
    """The index level the options imply (put-call parity on the five strikes nearest the money). NaN if it can't be found."""
    p = chain[chain["side"] == "P"].set_index("strike")
    c = chain[chain["side"] == "C"].set_index("strike")
    both = p.join(c, lsuffix="_p", rsuffix="_c", how="inner")
    both = both[(both["bid_p"] > 0) & (both["bid_c"] > 0)]
    if len(both) < 3:
        return float("nan")
    near = (both["mid_c"] - both["mid_p"]).abs().nsmallest(5).index
    return float(np.median(near.to_numpy() + math.exp(r * T) * (both.loc[near, "mid_c"] - both.loc[near, "mid_p"]).to_numpy()))


def _as_put(side, mid, K, F, T, r):
    """A call's price turned into the matching put's price (put-call parity), so one formula handles both sides."""
    return np.where(side == "P", mid, mid - math.exp(-r * T) * (F - K))


def smile_table(chain: pd.DataFrame, F: float, T: float, r: float) -> pd.DataFrame:
    """Out-of-the-money options on both sides of the index: implied volatility, and what the prices imply about where
    the index finishes. `p_below` is the chance the options put on finishing below each strike. It includes the tilt of
    the smile (lower strikes trade at higher volatility), which moves a 20-delta put's chance by several points."""
    o = chain[(chain["bid"] > 0) & (((chain["side"] == "P") & (chain["strike"] <= F)) | ((chain["side"] == "C") & (chain["strike"] > F)))]
    o = o.sort_values("strike").reset_index(drop=True)
    cols = ["strike", "side", "bid", "ask", "mid", "iv", "p_below"]
    if len(o) < 3:
        return pd.DataFrame(columns=cols)
    K = o["strike"].to_numpy(float)
    o["iv"] = put_iv(_as_put(o["side"].to_numpy(), o["mid"].to_numpy(float), K, F, T, r), F, K, T, r)
    o = o.dropna(subset=["iv"]).reset_index(drop=True)
    if len(o) < 3:
        return pd.DataFrame(columns=cols)
    K, iv = o["strike"].to_numpy(float), o["iv"].to_numpy(float)
    smooth = pd.Series(iv).rolling(5, center=True, min_periods=1).mean().to_numpy()
    slope = np.gradient(smooth, K)                               # how fast volatility changes from strike to strike
    sd = iv * math.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sd * sd) / sd
    raw = np.clip(ndtr(-(d1 - sd)) + F * np.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi) * math.sqrt(T) * slope, 0.0, 1.0)
    # a chance of finishing below a strike can only rise with the strike; iron out the noise from wide quotes
    o["p_below"] = 0.5 * (np.maximum.accumulate(raw) + np.minimum.accumulate(raw[::-1])[::-1])
    return o[cols]


def puts_table(chain: pd.DataFrame, F: float, T: float, r: float, smile: pd.DataFrame | None = None) -> pd.DataFrame:
    """The puts below the index with implied volatility, delta, and the options' chance of finishing above each strike."""
    p = chain[(chain["side"] == "P") & (chain["strike"] < F)].copy()
    if not len(p):
        return p.assign(iv=[], delta=[], below=[], p_above=[], rel_spread=[])
    K = p["strike"].to_numpy(float)
    p["iv"] = put_iv(p["mid"].to_numpy(float), F, K, T, r)
    p = p.dropna(subset=["iv"])
    K, iv = p["strike"].to_numpy(float), p["iv"].to_numpy(float)
    sd = iv * math.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sd * sd) / sd
    p["delta"] = -math.exp(-r * T) * ndtr(-d1)
    plain = ndtr(-(d1 - sd))                                     # chance below the strike with no allowance for the smile
    smile = smile_table(chain, F, T, r) if smile is None else smile
    if len(smile) >= 3:
        sk, sp = smile["strike"].to_numpy(float), smile["p_below"].to_numpy(float)
        below = np.where(K < sk[0], np.minimum(plain, sp[0]), np.interp(K, sk, sp))
    else:
        below = plain
    p["p_above"] = 1 - below
    p["below"] = 1 - K / F
    p["rel_spread"] = (p["ask"] - p["bid"]) / p["mid"]
    return p.sort_values("strike").reset_index(drop=True)


def atm_vol(chain: pd.DataFrame, F: float, T: float, r: float) -> float:
    """Implied volatility at the money: the average of the nearest put and call."""
    out = []
    for side in ("P", "C"):
        s = chain[(chain["side"] == side) & (chain["bid"] > 0)]
        if not len(s):
            continue
        row = s.iloc[(s["strike"] - F).abs().argmin()]
        K = float(row["strike"])
        v = put_iv(_as_put(np.array([side]), np.array([float(row["mid"])]), np.array([K]), F, T, r), F, [K], T, r)[0]
        if v == v:
            out.append(float(v))
    return float(np.mean(out)) if out else float("nan")


QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)


def implied_range(smile: pd.DataFrame, F: float, T: float, vol: float) -> dict:
    """Where the options put the index at expiry: the middle half and the middle 80%, as moves from today's level.
    Read off the smile when it reaches far enough; otherwise from at-the-money volatility alone (`from_smile` False)."""
    move = vol * math.sqrt(T)
    out = dict(move=move, from_smile=False)
    if len(smile) >= 5 and smile["p_below"].iloc[0] <= 0.10 and smile["p_below"].iloc[-1] >= 0.90:
        k = np.interp(QUANTILES, smile["p_below"].to_numpy(float), smile["strike"].to_numpy(float))
        out.update(from_smile=True, **{f"q{int(q * 100)}": float(x / F - 1) for q, x in zip(QUANTILES, k)})
    else:
        from scipy.special import ndtri
        out.update({f"q{int(q * 100)}": float(math.exp(ndtri(q) * move - 0.5 * move * move) - 1) for q in QUANTILES})
    return out


# ---------------------------------------------------------------- spreads
def spread_numbers(short, long_, fill="quote", commission=1.0) -> dict:
    """Credit, worst case and breakeven of one put credit spread (per spread, in dollars)."""
    if fill == "mid":
        credit = float(short["mid"] - long_["mid"])
    else:
        credit = float(short["bid"] - long_["ask"])            # sell at the bid, buy at the ask
    width = float(short["strike"] - long_["strike"])
    net = credit * MULTIPLIER - 2 * commission
    return dict(credit=credit, width=width, net=net, risk=(width - credit) * MULTIPLIER + 2 * commission,
                breakeven=float(short["strike"]) - net / MULTIPLIER)


def _usable_puts(puts, max_rel_spread, min_bid):
    return puts[(puts["bid"] >= min_bid) & (puts["rel_spread"] <= max_rel_spread)]


def find_spreads(puts: pd.DataFrame, target: float, cap: float, *, commission=1.0, fill="quote", max_short_delta=0.30,
                 max_rel_spread=0.60, min_bid=0.05, max_contracts=500) -> tuple[pd.DataFrame, dict]:
    """Put credit spreads that collect at least `target` dollars without risking more than `cap` dollars.

    For every short strike it keeps the long strike that risks the least. The result is sorted safest first: the short
    strike furthest below the index has the best chance of the whole premium being kept.
    Returns (table, info). If nothing reaches the target, the table is empty and info says how much the cap allows."""
    shorts = _usable_puts(puts, max_rel_spread, min_bid)
    shorts = shorts[shorts["delta"].abs() <= max_short_delta]
    longs = puts[puts["ask"] > 0]
    rows, most = [], None
    for _, s in shorts.iterrows():
        best = None
        for _, l in longs[longs["strike"] < s["strike"]].iterrows():
            x = spread_numbers(s, l, fill, commission)
            if x["net"] <= 0 or x["risk"] <= 0:
                continue
            fit = int(cap // x["risk"])                         # the most spreads the loss cap allows
            if fit >= 1 and (most is None or fit * x["net"] > most["premium"]):
                most = dict(premium=fit * x["net"], n=fit, short=float(s["strike"]), long=float(l["strike"]), delta=float(s["delta"]))
            n = int(math.ceil(target / x["net"]))
            if n < 1 or n > max_contracts or n * x["risk"] > cap + 1e-9:
                continue
            if best is None or n * x["risk"] < best["max_loss"]:
                best = dict(short=float(s["strike"]), long=float(l["strike"]), n=n, credit=x["credit"], width=x["width"],
                            premium=n * x["net"], max_loss=n * x["risk"], breakeven=x["breakeven"], delta=float(s["delta"]),
                            below=float(s["below"]), p_above=float(s["p_above"]), iv=float(s["iv"]),
                            short_bid=float(s["bid"]), short_ask=float(s["ask"]), long_bid=float(l["bid"]), long_ask=float(l["ask"]))
        if best:
            rows.append(best)
    table = pd.DataFrame(rows).sort_values("short").reset_index(drop=True) if rows else pd.DataFrame()
    return table, dict(most_within_cap=most, shorts_considered=int(len(shorts)))


def tested_spread(puts: pd.DataFrame, cap: float, *, commission=1.0, fill="quote", max_rel_spread=0.60) -> dict | None:
    """The spread the backtests traded: sell the put nearest 20 delta, buy the put nearest 10 delta, sized to the loss cap."""
    shorts = puts[(puts["bid"] > 0) & (puts["rel_spread"] <= max_rel_spread)]
    if not len(shorts):
        return None
    s = shorts.iloc[(shorts["delta"] + TESTED_SHORT_DELTA).abs().argmin()]
    if not 0.12 <= -s["delta"] <= 0.30:
        return None
    longs = puts[(puts["strike"] < s["strike"]) & (puts["ask"] > 0)]
    if not len(longs):
        return None
    l = longs.iloc[(longs["delta"] + TESTED_LONG_DELTA).abs().argmin()]
    if not 0.03 <= -l["delta"] <= 0.17:
        return None
    x = spread_numbers(s, l, fill, commission)
    if x["net"] <= 0 or x["risk"] <= 0:
        return None
    n = int(cap // x["risk"])
    return dict(short=float(s["strike"]), long=float(l["strike"]), n=n, credit=x["credit"], width=x["width"], premium=n * x["net"],
                max_loss=n * x["risk"], breakeven=x["breakeven"], delta=float(s["delta"]), long_delta=float(l["delta"]),
                below=float(s["below"]), p_above=float(s["p_above"]), iv=float(s["iv"]), net_each=x["net"], risk_each=x["risk"],
                short_bid=float(s["bid"]), short_ask=float(s["ask"]), long_bid=float(l["bid"]), long_ask=float(l["ask"]))


# ---------------------------------------------------------------- the range model
FEATURES = ["vxn", "rv5", "rv20", "rv60"]


def model_inputs(close: pd.Series, vxn: pd.Series) -> pd.DataFrame:
    """Daily inputs (in logs): VXN, and how much the index actually moved over the last 5, 20 and 60 days."""
    lr = np.log(close).diff()
    f = pd.DataFrame({"vxn": np.log(vxn / 100.0)}, index=close.index)
    for n in (5, 20, 60):
        f[f"rv{n}"] = np.log(lr.rolling(n).std() * math.sqrt(252))
    return f.replace([np.inf, -np.inf], np.nan)


def _ols(X, y):
    return np.linalg.lstsq(np.column_stack([np.ones(len(X)), X]), y, rcond=None)[0]


def _apply(b, X):
    return np.exp(np.column_stack([np.ones(len(X)), X]) @ b)


def fit_range_model(close: pd.Series, vxn: pd.Series, h: int, first_year: int | None = None) -> dict:
    """A forecast of how much the index moves over the next `h` trading days.

    Step 1: predict the size of the move (annualised volatility) from VXN and recent actual movement.
    Step 2: divide every past h-day return by the size predicted for it at the time. Those scaled returns give the shape
            of the range (fat tails, bigger falls than rises), with no bell-curve assumption.
    Both steps are checked walk-forward: every year is predicted by a fit that only saw earlier, finished windows."""
    close = close.dropna()
    vxn = vxn.reindex(close.index).ffill(limit=5)
    lr = np.log(close).diff()
    f = model_inputs(close, vxn)
    hv = max(h, 5)                                               # the size of daily moves is measured over at least a week
    d = f.assign(ret=np.log(close.shift(-h) / close),
                 y=np.log(np.sqrt((lr ** 2).rolling(hv).sum().shift(-hv) * 252.0 / hv))).replace([np.inf, -np.inf], np.nan)
    known = d.dropna()                                           # windows that have finished
    if len(known) < 1500:
        raise ValueError("not enough history to fit the range model")
    years = sorted(set(known.index.year))
    first_year = first_year or years[0] + 5
    root_t = math.sqrt(h / 252.0)
    parts = []
    for year in [y for y in years if y >= first_year]:
        cut = pd.Timestamp(year, 1, 1) - pd.Timedelta(days=int(hv * 1.6) + 3)       # only windows finished before the year starts
        tr, te = known[known.index < cut], known[known.index.year == year]
        if len(tr) < 750 or not len(te):
            continue
        b = _ols(tr[FEATURES].to_numpy(), tr["y"].to_numpy())
        z_tr = tr["ret"].to_numpy() / (_apply(b, tr[FEATURES].to_numpy()) * root_t)
        sig = _apply(b, te[FEATURES].to_numpy())
        q10, q90 = np.quantile(z_tr, [0.10, 0.90])
        parts.append(pd.DataFrame({"ret": te["ret"].to_numpy(), "sig": sig, "z": te["ret"].to_numpy() / (sig * root_t),
                                   "lo": q10 * sig * root_t, "hi": q90 * sig * root_t, "rv": np.exp(te["y"].to_numpy()),
                                   "vxn": np.exp(te["vxn"].to_numpy())}, index=te.index))
    oos = pd.concat(parts)
    b = _ols(known[FEATURES].to_numpy(), known["y"].to_numpy())
    ratio = oos["rv"] / oos["sig"]
    lo_v, hi_v = -Z80 * oos["vxn"] * root_t, Z80 * oos["vxn"] * root_t
    check = {"years": f"{oos.index.year.min()}-{oos.index.year.max()}", "windows": int(len(oos)),
             "inside": float(((oos["ret"] >= oos["lo"]) & (oos["ret"] <= oos["hi"])).mean()),
             "below": float((oos["ret"] < oos["lo"]).mean()), "above": float((oos["ret"] > oos["hi"]).mean()),
             "inside_vxn": float(((oos["ret"] >= lo_v) & (oos["ret"] <= hi_v)).mean()),
             "below_vxn": float((oos["ret"] < lo_v).mean()),
             "size_low": float(ratio.quantile(0.10)), "size_high": float(ratio.quantile(0.90)),
             "vxn_overstates": float(np.exp(np.mean(np.log(oos["vxn"] / oos["rv"]))) - 1),
             "error_model": float(np.mean(np.abs(np.log(oos["rv"] / oos["sig"])))),
             "error_vxn": float(np.mean(np.abs(np.log(oos["rv"] / oos["vxn"]))))}
    latest = f.dropna().iloc[-1]
    return dict(h=h, coef=b, z=np.sort(oos["z"].to_numpy()), n_eff=max(len(oos) / h, 1.0), check=check,
                z_rms=float(np.sqrt(np.mean(oos["z"].to_numpy() ** 2))), oos=oos[["ret", "sig", "z", "lo", "hi"]],
                sigma=float(_apply(b, latest[FEATURES].to_numpy()[None, :])[0]), as_of=f.dropna().index[-1],
                inputs={k: float(math.exp(latest[k])) for k in FEATURES})


def model_range(model: dict) -> dict:
    """The model's range for the index at expiry, as moves from today's level: the middle half, the middle 80%, the
    typical size of the move, and how far off that size has usually been (the model's error bar)."""
    scale = model["sigma"] * math.sqrt(model["h"] / 252.0)
    out = {f"q{int(q * 100)}": float(math.exp(x * scale) - 1) for q, x in zip(QUANTILES, np.quantile(model["z"], QUANTILES))}
    move = scale * model["z_rms"]
    out.update(move=move, move_low=move * model["check"]["size_low"], move_high=move * model["check"]["size_high"])
    return out


def wilson(p: float, n: float, z: float = Z80) -> tuple[float, float]:
    """A range for a share measured on about n independent cases (80% by default)."""
    if n <= 0:
        return 0.0, 1.0
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(centre - half, 0.0), min(centre + half, 1.0)


def chance_above(model: dict, level: float, strike: float) -> dict:
    """How often, in the model's history, the index would have finished above `strike`, with an 80% range.
    The range allows for overlapping windows by counting about one independent case per holding period."""
    scale = model["sigma"] * math.sqrt(model["h"] / 252.0)
    z_k = math.log(strike / level) / scale
    p_below = float(np.searchsorted(model["z"], z_k, side="right")) / len(model["z"])
    lo, hi = wilson(p_below, model["n_eff"])
    return dict(p=1 - p_below, lo=1 - hi, hi=1 - lo)


def model_value(model: dict, level: float, spread: dict, commission=1.0) -> dict:
    """What the spread would have made per dollar at risk, on average, across the model's history of scaled moves."""
    scale = model["sigma"] * math.sqrt(model["h"] / 252.0)
    end = level * np.exp(model["z"] * scale)
    paid = np.maximum(spread["short"] - end, 0.0) - np.maximum(spread["long"] - end, 0.0)
    net = spread["credit"] * MULTIPLIER - 2 * commission
    risk = (spread["width"] - spread["credit"]) * MULTIPLIER + 2 * commission
    per = (net - paid * MULTIPLIER) / risk
    se = float(per.std(ddof=1) / math.sqrt(model["n_eff"]))
    mean = float(per.mean())
    return dict(mean=mean, lo=mean - Z80 * se, hi=mean + Z80 * se, win=float((per > 0).mean()), full_loss=float((per <= -0.999).mean()))


# ---------------------------------------------------------------- the rule and the recommendation
def trend_state(close: pd.Series, days: int = TREND_DAYS) -> dict:
    """Where the last close sits against its 200-day average (the rule checks the night before entry)."""
    c = close.dropna()
    avg = float(c.iloc[-days:].mean())
    last = float(c.iloc[-1])
    return dict(last=last, average=avg, gap=last / avg - 1, above=last >= avg, as_of=c.index[-1], days=days)


def recommend(horizon: str, trend: dict, spread: dict | None, value: dict | None, days_left: int | None = None) -> dict:
    """The recommendation for one horizon ("week" or "month"), in plain words. It mirrors what the backtests support:
    a 25-day spread only when the index is above its 200-day average; weekly spreads lost in every period tested, so
    the default there is to stand aside. The model's view is added as context, because it was never tested as a signal."""
    notes = []
    if spread is None or spread.get("n", 0) < 1:
        why = ("One spread risks more than your loss cap allows." if spread is not None
               else "The quotes are too thin to build the tested spread (20-delta put sold, 10-delta put bought).")
        return dict(action="stand aside", headline="No trade: the tested spread can't be built within your loss cap.", notes=[why])
    if value is not None:
        if value["hi"] < 0:
            notes.append("Model's view: the premium looks thin for the range it expects. Its estimate is below zero even at the top of its range.")
        elif value["lo"] > 0:
            notes.append("Model's view: the premium looks fair or better. Its estimate is above zero across its whole range.")
        else:
            notes.append("Model's view: too close to call. Its estimate straddles zero, which is the usual case.")
    if horizon == "week":
        notes.insert(0, "Weekly spreads lost money in every period tested (2001-19 on estimated prices, 2020-26 on real quotes). "
                        "Costs take about a fifth of the premium.")
        return dict(action="stand aside", headline="Stand aside this week. The spread below is for reference only.", notes=notes)
    if not trend["above"]:
        notes.insert(0, f"The index last closed {abs(trend['gap']):.1%} below its {trend['days']}-day average. "
                        "In 2020-26 the big losses came in months that started this way.")
        return dict(action="stand aside", headline="Stand aside this month: the 200-day rule says no new spread.", notes=notes)
    notes.insert(0, f"The index last closed {trend['gap']:.1%} above its {trend['days']}-day average, so the 200-day rule allows a spread.")
    if days_left is not None and not 18 <= days_left <= 35:
        notes.append(f"This expiry is {days_left} days away. The backtests opened spreads about 25 days out, so this one is outside what was tested.")
    notes.append("One spread at a time: the loss cap is for everything open. If last month's spread is still on, this is not one to add.")
    notes.append("The evidence is thin. The rule beat plain QQQ after tax in 2020-26 but not in 2001-19 on estimated prices. Paper-trade it or size small.")
    return dict(action="trade", headline="The 200-day rule allows this month's spread.", notes=notes)


# ---------------------------------------------------------------- which expirations
def third_friday(year: int, month: int) -> dt.date:
    first = dt.date(year, month, 1)
    return first + dt.timedelta(days=(4 - first.weekday()) % 7 + 14)


def week_ends(expirations, today: dt.date) -> list[dt.date]:
    """The last listed expiration of each coming week (Friday, or Thursday when Friday is a holiday)."""
    weeks = {}
    for x in sorted({e for e in expirations if e >= today}):
        weeks.setdefault(x.isocalendar()[:2], []).append(x)
    return [max(v) for _, v in sorted(weeks.items())]


def pick_weekly(ends: list[dt.date], now: pd.Timestamp) -> int:
    """This week's expiration Monday to Thursday; next week's from Friday on (a spread opened on expiry day is a coin flip)."""
    if len(ends) > 1 and (ends[0] - now.date()).days < 1:
        return 1
    if len(ends) > 1 and now.weekday() >= 4 and ends[0].isocalendar()[:2] == now.date().isocalendar()[:2]:
        return 1
    return 0


def pick_monthly(ends: list[dt.date], today: dt.date, target_days: int = 25) -> int:
    """The week-ending expiration closest to 25 days away, which is what the backtests held."""
    if not ends:
        return 0
    return int(np.argmin([abs((e - today).days - target_days) + (1000 if (e - today).days < 10 else 0) for e in ends]))


def is_monthly(expiration: dt.date) -> bool:
    """True for the standard monthly expiration (the third Friday, or the Thursday before it in a holiday week)."""
    tf = third_friday(expiration.year, expiration.month)
    return expiration == tf or (expiration == tf - dt.timedelta(days=1) and expiration.weekday() == 3)
