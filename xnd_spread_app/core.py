"""Logic for the XND put spread planner. No network and no Streamlit in here, so every function can be tested on its own.

Three parts:
  1. option maths and cleaning one expiration's quotes,
  2. the spread finder (best chance to collect a target premium within a loss cap) and the standard 20-delta / 10-delta spread,
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
TESTED_SHORT_DELTA, TESTED_LONG_DELTA = 0.20, 0.10      # the standard spread: sell the 20-delta put, buy the 10-delta put
TREND_DAYS = 200               # the monthly spread is only opened when the index closed above this average
VALUE_SELL_DELTAS = (0.10, 0.30)   # best value: the put sold must be between these deltas
VALUE_WIDTHS = (0.01, 0.06)        # best value: the two strikes must be this far apart, as a share of the index


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
            fit = int(cap // x["risk"]) if math.isfinite(cap) else 0      # the most spreads the loss cap allows
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


def risk_needed(puts: pd.DataFrame, target: float, **kw) -> pd.DataFrame:
    """Every way to collect `target` when the loss cap is ignored, least at risk first. The first row is the smallest
    loss cap at which the target can be reached at all. Empty if these quotes can't reach it at any risk level."""
    table, _ = find_spreads(puts, target, math.inf, **kw)
    return table.sort_values(["max_loss", "short"]).reset_index(drop=True) if len(table) else table


def tested_spread(puts: pd.DataFrame, cap: float, *, commission=1.0, fill="quote", max_rel_spread=0.60) -> tuple[dict | None, str | None]:
    """The standard spread: sell the put nearest 20 delta, buy the put nearest 10 delta, sized to the loss cap.
    Returns (spread, None), or (None, why it can't be built from these quotes)."""
    shorts = puts[(puts["bid"] > 0) & (puts["rel_spread"] <= max_rel_spread)]
    if not len(shorts):
        return None, "No put has a usable quote: every bid is zero or the gap between bid and ask is wider than your filter allows."
    s = shorts.iloc[(shorts["delta"] + TESTED_SHORT_DELTA).abs().argmin()]
    if not 0.12 <= -s["delta"] <= 0.30:
        return None, (f"No put near 20 delta has a usable quote. The closest is the {s['strike']:g} put at {abs(s['delta']):.2f} delta; "
                      "the ones nearer 20 have no bid or a bid-ask gap wider than your filter allows.")
    longs = puts[(puts["strike"] < s["strike"]) & (puts["ask"] > 0)]
    if not len(longs):
        return None, f"There is no put quoted below the {s['strike']:g} put to buy as protection."
    l = longs.iloc[(longs["delta"] + TESTED_LONG_DELTA).abs().argmin()]
    if not 0.03 <= -l["delta"] <= 0.17:
        return None, f"No put near 10 delta is quoted to buy as protection. The closest is the {l['strike']:g} put at {abs(l['delta']):.2f} delta."
    x = spread_numbers(s, l, fill, commission)
    if x["net"] <= 0 or x["risk"] <= 0:
        if fill == "quote":
            return None, (f"At these quotes the spread pays nothing after costs: the {s['strike']:g} put is bid {s['bid']:.2f} and the "
                          f"{l['strike']:g} put is offered at {l['ask']:.2f}. Gaps this wide between bid and ask are usual outside market hours.")
        return None, f"The {s['strike']:g} / {l['strike']:g} spread pays nothing after costs at these prices."
    n = int(cap // x["risk"])
    return dict(short=float(s["strike"]), long=float(l["strike"]), n=n, credit=x["credit"], width=x["width"], premium=n * x["net"],
                max_loss=n * x["risk"], breakeven=x["breakeven"], delta=float(s["delta"]), long_delta=float(l["delta"]),
                below=float(s["below"]), p_above=float(s["p_above"]), iv=float(s["iv"]), net_each=x["net"], risk_each=x["risk"],
                short_bid=float(s["bid"]), short_ask=float(s["ask"]), long_bid=float(l["bid"]), long_ask=float(l["ask"])), None


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


def fit_range_model(close: pd.Series, vxn: pd.Series, h: int, first_year: int | None = None, pick_year: int | None = None) -> dict:
    """A forecast of how much the index moves over the next `h` trading days.

    Step 1: predict the size of the move (annualised volatility) from VXN and recent actual movement.
    Step 2: divide every past h-day return by the size predicted for it at the time. Those scaled returns give the shape
            of the range (fat tails, bigger falls than rises), with no bell-curve assumption.
    Both steps are checked walk-forward: every year is predicted by a fit that only saw earlier, finished windows.

    The result also holds `pick`: the fit used to choose strikes, made exactly as in the Phase 26 backtest. It is refitted
    once a year, on windows that ended before `pick_year` began (this year, unless given), and its scaled moves are the
    ones from those same windows."""
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
    pick_year = pick_year or (close.index[-1] + pd.offsets.BDay(1)).year
    where = pd.Series(np.arange(len(close)), index=close.index).reindex(known.index).to_numpy()
    tr = known[close.index[where + hv] < pd.Timestamp(pick_year, 1, 1)]       # the window's last day is before the year began
    if len(tr) < 750:
        tr = known
    b_pick = _ols(tr[FEATURES].to_numpy(), tr["y"].to_numpy())
    z_pick = np.sort(tr["ret"].to_numpy() / (_apply(b_pick, tr[FEATURES].to_numpy()) * root_t))
    pick = dict(coef=b_pick, z=z_pick, n_eff=max(len(z_pick) / h, 1.0), windows=int(len(tr)), year=int(pick_year),
                sigma=float(_apply(b_pick, latest[FEATURES].to_numpy()[None, :])[0]))
    return dict(h=h, coef=b, z=np.sort(oos["z"].to_numpy()), n_eff=max(len(oos) / h, 1.0), check=check, pick=pick,
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


def _view(model: dict, use: str = "main") -> tuple[float, np.ndarray, float]:
    """(size of the move over the holding period, past scaled moves, about how many independent cases those are).
    use="main" is the walk-forward model behind the range chart; use="pick" is the yearly refit that chooses strikes."""
    m = model["pick"] if use == "pick" else model
    return m["sigma"] * math.sqrt(model["h"] / 252.0), m["z"], m["n_eff"]


def chance_above(model: dict, level: float, strike: float, use: str = "main") -> dict:
    """How often, in the model's history, the index would have finished above `strike`, with an 80% range.
    The range allows for overlapping windows by counting about one independent case per holding period."""
    scale, z, n_eff = _view(model, use)
    z_k = math.log(strike / level) / scale
    p_below = float(np.searchsorted(z, z_k, side="right")) / len(z)
    lo, hi = wilson(p_below, n_eff)
    return dict(p=1 - p_below, lo=1 - hi, hi=1 - lo)


def model_value(model: dict, level: float, spread: dict, commission=1.0, use: str = "main") -> dict:
    """What the spread would have made per dollar at risk, on average, across the model's history of scaled moves."""
    scale, z, n_eff = _view(model, use)
    end = level * np.exp(z * scale)
    paid = np.maximum(spread["short"] - end, 0.0) - np.maximum(spread["long"] - end, 0.0)
    net = spread["credit"] * MULTIPLIER - 2 * commission
    risk = (spread["width"] - spread["credit"]) * MULTIPLIER + 2 * commission
    per = (net - paid * MULTIPLIER) / risk
    se = float(per.std(ddof=1) / math.sqrt(n_eff))
    mean = float(per.mean())
    return dict(mean=mean, lo=mean - Z80 * se, hi=mean + Z80 * se, win=float((per > 0).mean()), full_loss=float((per <= -0.999).mean()))


def best_value_spread(puts: pd.DataFrame, model: dict, level: float, cap: float, *, commission=1.0, fill="quote",
                      max_short_delta=VALUE_SELL_DELTAS[1], max_rel_spread=0.60) -> tuple[dict | None, str | None]:
    """The "model best value" spread from the Phase 26 backtest: of every allowed pair of puts, the one the model values
    most (expected profit per dollar at risk, after costs), sized to the loss cap.

    Allowed: both puts have a bid and a bid-ask gap inside the filter; the put sold is between 10 delta and
    `max_short_delta`; the strikes are 1% to 6% of the index apart; the pair pays something after commission.
    Returns (spread, None), or (None, why no pair qualifies)."""
    ok = puts[(puts["bid"] > 0) & (puts["rel_spread"] <= max_rel_spread)].reset_index(drop=True)
    if len(ok) < 2:
        return None, "Fewer than two puts have a usable quote (a bid above zero and a bid-ask gap inside your filter)."
    scale, z, _ = _view(model, "pick")
    K, bid, ask, mid, delta = (ok[c].to_numpy(float) for c in ("strike", "bid", "ask", "mid", "delta"))
    end = level * np.exp(z * scale)
    pay = np.maximum(K[:, None] - end[None, :], 0.0).mean(axis=1)          # what each put pays on average across the model's past moves
    credit = (mid[:, None] - mid[None, :]) if fill == "mid" else (bid[:, None] - ask[None, :])     # row = put sold, column = put bought
    width = K[:, None] - K[None, :]
    net = credit * MULTIPLIER - 2 * commission
    risk = (width - credit) * MULTIPLIER + 2 * commission
    sellable = (-delta >= VALUE_SELL_DELTAS[0]) & (-delta <= max_short_delta)
    if not sellable.any():
        return None, f"No put between {VALUE_SELL_DELTAS[0]:.2f} and {max_short_delta:.2f} delta has a usable quote."
    allowed = (width >= VALUE_WIDTHS[0] * level) & (width <= VALUE_WIDTHS[1] * level) & (net > 0) & (risk > 0) & sellable[:, None]
    if not allowed.any():
        return None, ("No allowed pair of puts pays anything after costs at these quotes. "
                      "Gaps this wide between bid and ask are usual outside market hours.")
    value = np.where(allowed, (net - MULTIPLIER * (pay[:, None] - pay[None, :])) / np.where(risk > 0, risk, np.nan), -np.inf)
    i, j = np.unravel_index(int(np.argmax(value)), value.shape)
    sh, lg = ok.loc[i], ok.loc[j]
    x = spread_numbers(sh, lg, fill, commission)
    n = int(cap // x["risk"])
    return dict(short=float(sh["strike"]), long=float(lg["strike"]), n=n, credit=x["credit"], width=x["width"], premium=n * x["net"],
                max_loss=n * x["risk"], breakeven=x["breakeven"], delta=float(sh["delta"]), long_delta=float(lg["delta"]),
                below=float(sh["below"]), p_above=float(sh["p_above"]), iv=float(sh["iv"]), net_each=x["net"], risk_each=x["risk"],
                short_bid=float(sh["bid"]), short_ask=float(sh["ask"]), long_bid=float(lg["bid"]), long_ask=float(lg["ask"]),
                model_value=float(value[i, j]), pairs_considered=int(allowed.sum())), None


# ---------------------------------------------------------------- the rule and the recommendation
def trend_state(close: pd.Series, days: int = TREND_DAYS) -> dict:
    """Where the last close sits against its 200-day average (the rule checks the night before entry)."""
    c = close.dropna()
    avg = float(c.iloc[-days:].mean())
    last = float(c.iloc[-1])
    return dict(last=last, average=avg, gap=last / avg - 1, above=last >= avg, as_of=c.index[-1], days=days)


def recommend(horizon: str, trend: dict, spread: dict | None, value: dict | None, days_left: int | None = None,
              why: str | None = None) -> dict:
    """The recommendation for one horizon ("week" or "month"), in plain words.

    Both horizons get the model's view of the premium: what the options pay against the move the model expects.
    The monthly spread also follows the 200-day rule (no new spread when the index closed below its 200-day average).
    Returns action ("trade" or "stand aside"), tone ("good", "neutral" or "caution"), a headline and notes."""
    if spread is None:
        return dict(action="stand aside", tone="caution", headline="No trade: the spread can't be built from these quotes.",
                    notes=[why or "The quotes are too thin to build a spread."])
    if spread.get("n", 0) < 1:
        return dict(action="stand aside", tone="caution", headline="No trade: one spread risks more than your loss cap.",
                    notes=["Raise the loss cap to fit at least one spread."])
    period = "week" if horizon == "week" else "month"
    if value is None:
        view, tone, model_note = "", "neutral", None
    elif value["mean"] <= 0:
        view, tone = "even the best pair is not worth selling to the model", "caution"
        model_note = ("Model's view: of every allowed pair of puts, the best is still worth zero or less per dollar at risk. "
                      "The backtest's stricter version of this rule skipped trades like this one.")
    elif value["hi"] < 0:
        view, tone = "the premium looks thin to the model", "caution"
        model_note = "Model's view: the premium is thin for the move it expects. Its estimate of the spread's value is below zero even at the top of its range."
    elif value["lo"] > 0:
        view, tone = "the model sees more premium than the move it expects", "good"
        model_note = "Model's view: options are paying for more movement than it expects. Its estimate of the spread's value is above zero across its whole range."
    else:
        view, tone = "the premium looks about fair to the model", "neutral"
        model_note = "Model's view: about fair. Its estimate of the spread's value straddles zero, which is the usual case."
    notes = [model_note] if model_note else []
    if horizon == "month":
        if not trend["above"]:
            notes.insert(0, f"The index last closed {abs(trend['gap']):.1%} below its {trend['days']}-day average.")
            return dict(action="stand aside", tone="caution", headline="Stand aside this month: the 200-day rule says no new spread.", notes=notes)
        notes.insert(0, f"The index last closed {trend['gap']:.1%} above its {trend['days']}-day average, so the 200-day rule allows a spread.")
        if days_left is not None and not 18 <= days_left <= 35:
            notes.append(f"This expiry is {days_left} days away. The monthly spread is normally opened about 25 days out.")
        headline = "The 200-day rule allows this month's spread" + (f", but {view}." if tone == "caution" else ".")
    else:
        headline = f"This week's spread: {view}." if view else "This week's spread."
        if not trend["above"]:
            notes.append(f"The index last closed {abs(trend['gap']):.1%} below its {trend['days']}-day average. The 200-day rule is only applied to the monthly spread.")
    notes.append(f"The loss cap is for everything open. If an earlier spread for this {period} is still on, this is not one to add on top.")
    return dict(action="trade", tone=tone, headline=headline, notes=notes)


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
    """The week-ending expiration closest to 25 days away."""
    if not ends:
        return 0
    return int(np.argmin([abs((e - today).days - target_days) + (1000 if (e - today).days < 10 else 0) for e in ends]))


def is_monthly(expiration: dt.date) -> bool:
    """True for the standard monthly expiration (the third Friday, or the Thursday before it in a holiday week)."""
    tf = third_friday(expiration.year, expiration.month)
    return expiration == tf or (expiration == tf - dt.timedelta(days=1) and expiration.weekday() == 3)
