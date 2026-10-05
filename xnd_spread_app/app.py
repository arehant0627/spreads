"""XND put spread planner.

For this week and for the month ahead it shows
  1. how far the options say the Nasdaq-100 could move, next to what a model built on 25 years of history expects,
  2. the put credit spread on XND (the Nasdaq-100 micro index) with the best chance of collecting a premium you choose,
  3. the trade the backtests support, or a plain "stand aside".

Run with:  streamlit run app.py
The ThetaData key is read from Streamlit Secrets (THETADATA_API_KEY). It is never written in this file."""
import inspect

import altair as alt
import pandas as pd
import streamlit as st

import core
import data

st.set_page_config(page_title="XND put spread planner", layout="wide")


# ---------------------------------------------------------------- small helpers
def usd(v, cents=False):
    s = f"{abs(v):,.2f}" if cents else f"{abs(v):,.0f}"
    return f"−${s}" if v < 0 else f"${s}"


def esc(text):
    """Dollar signs in running text would otherwise be read as the start of a formula."""
    return text.replace("$", "\\$")


def pct(v, d=1):
    return f"{v:.{d}%}".replace("-", "−")


def signed(v, d=1):
    return ("+" if v >= 0 else "−") + f"{abs(v):.{d}%}"


def day(d):
    return f"{d:%a %b} {d.day}"


def cents(v):
    """A value per dollar at risk, in cents."""
    return ("+" if v >= 0 else "−") + f"{abs(v) * 100:.1f}¢"


_ARROW_OFF = {"delta_arrow": "off"} if "delta_arrow" in inspect.signature(st.metric).parameters else {}


def metric(col, label, value, sub=None, help=None):
    """A metric with an optional grey note underneath (no up or down arrow)."""
    kw = {"help": help}
    if sub is not None:
        kw.update(delta=sub, delta_color="off", **_ARROW_OFF)
    col.metric(label, value, **kw)


def dark_mode() -> bool:
    try:
        return st.context.theme.type == "dark"
    except Exception:                                           # older Streamlit, or no browser attached
        return False


# ---------------------------------------------------------------- data source
try:
    API_KEY = st.secrets.get("THETADATA_API_KEY", "")
except Exception:                                               # no secrets file at all
    API_KEY = ""

st.title("XND put spread planner")
st.caption("Put credit spreads on XND, the Nasdaq-100 micro index (one hundredth of the index, cash-settled, moves in step "
           "with QQQ). A planning tool, not financial advice.")

sb = st.sidebar
demo = sb.toggle("Demo quotes (no ThetaData key needed)", value=not API_KEY,
                 help="Made-up option quotes built from the real index level and VXN. Turn off to use ThetaData quotes.")
if not demo and not API_KEY:
    sb.error("Add THETADATA_API_KEY to the app's secrets to use real quotes.")
    st.stop()

account = sb.number_input("Account value ($)", min_value=5_000, value=250_000, step=5_000)
sb.subheader("This week")
week_target = sb.number_input("Premium to collect this week ($)", min_value=10, value=125, step=25, key="week_target")
week_cap_pct = sb.number_input("Most you could lose this week (% of account)", min_value=0.1, max_value=20.0, value=0.75,
                               step=0.25, key="week_cap")
sb.subheader("This month")
month_target = sb.number_input("Premium to collect this month ($)", min_value=10, value=500, step=50, key="month_target")
month_cap_pct = sb.number_input("Most you could lose this month (% of account)", min_value=0.1, max_value=20.0, value=3.0,
                                step=0.5, key="month_cap",
                                help="The backtests put 6% of the account at risk in each monthly spread. Half of that is the default here.")
with sb.expander("Costs and filters"):
    commission = st.number_input("Commission per contract ($)", min_value=0.0, value=1.00, step=0.05,
                                 help=esc("The backtests assumed $1.00. Fidelity charges $0.65 plus small exchange fees on index options."))
    fill = st.radio("Prices used", ["quote", "mid"], format_func={"quote": "Sell at the bid, buy at the ask", "mid": "Halfway between bid and ask"}.get,
                    help="The first is what a market order gets and what the backtests assumed. The second is the best case for a limit order.")
    max_delta = st.slider("Closest strike to sell (delta)", 0.10, 0.40, 0.30, 0.01,
                          help="0.30 means roughly a 30% chance the index finishes below the strike you sell. Lower keeps you further away.")
    max_gap = st.slider("Widest bid-ask gap to accept (% of the price)", 10, 100, 60, 5) / 100
if sb.button("Refresh quotes", width="stretch"):
    st.cache_data.clear()


@st.cache_resource(show_spinner=False)
def client_for(key):
    return data.make_client(key)


@st.cache_data(ttl=3600, show_spinner=False)
def expirations_live(key):
    return data.list_expirations(client_for(key))


@st.cache_data(ttl=60, show_spinner=False)
def chain_live(key, expiration):
    return data.fetch_chain(client_for(key), expiration)


@st.cache_data(ttl=3600, show_spinner=False)
def history_live():
    return data.load_history()


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def rate_live():
    return data.tbill_rate()


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def model_for(h, last_day, _hist):
    return core.fit_range_model(_hist["ndx"], _hist["vxn"], h)


now = data.now_ny()
with st.spinner("Loading prices and quotes"):
    hist, hist_source = history_live()
    rate = rate_live()
    try:
        expirations = data.demo_expirations(now) if demo else expirations_live(API_KEY)
    except Exception as e:                                      # noqa: BLE001
        st.error(f"ThetaData didn't return the XND expirations ({type(e).__name__}: {str(e)[:200]}). Check the key in the app's secrets, or turn on demo quotes.")
        st.stop()

ends = core.week_ends(expirations, now.date())
if not ends:
    st.error("ThetaData returned no upcoming XND expirations.")
    st.stop()
trend = core.trend_state(hist["ndx"])
qqq_per_point = float(hist["qqq"].iloc[-1] / hist["ndx"].iloc[-1]) * 100        # QQQ dollars per XND point


def exp_label(e):
    d = (e - now.date()).days
    return f"{day(e)} · {d} day{'s' if d != 1 else ''}" + (" · monthly expiry" if core.is_monthly(e) else "")


week_choices = ends[:3]
month_choices = [e for e in ends if 10 <= (e - now.date()).days <= 60] or ends[-1:]
week_exp = sb.selectbox("Weekly expiry", week_choices, index=min(core.pick_weekly(week_choices, now), len(week_choices) - 1),
                        format_func=exp_label, key="week_exp")
month_exp = sb.selectbox("Monthly expiry", month_choices, index=core.pick_monthly(month_choices, now.date()),
                         format_func=exp_label, key="month_exp",
                         help="The backtests held spreads for about 25 days, so the expiry closest to 25 days away is picked for you.")


# ---------------------------------------------------------------- the work for one horizon
def analyse(horizon, expiration, target, cap):
    out = dict(horizon=horizon, expiration=expiration, target=target, cap=cap, days_left=(expiration - now.date()).days)
    if demo:
        quotes = data.demo_chain(float(hist["ndx"].iloc[-1]) / 100, float(hist["vxn"].iloc[-1]), expiration, now, rate)
        source, ts, notes = "demo", now, {"rows used": len(quotes)}
    else:
        try:
            quotes, source, ts, notes = chain_live(API_KEY, expiration)
        except Exception as e:                                  # noqa: BLE001
            return dict(out, error=f"ThetaData didn't return quotes for {day(expiration)} ({type(e).__name__}: {str(e)[:160]}).")
    out.update(source=source, ts=ts, notes=notes)
    chain = core.clean_chain(quotes, near=float(hist["ndx"].iloc[-1]) / 100)
    asof = ts if ts is not None else now
    T = core.years_to(expiration, asof)                         # time left on the options, from when they were quoted
    h = core.sessions_to(expiration, now)                       # trading sessions left from now, for the model
    F = core.forward_level(chain, rate, T)
    if F != F:
        return dict(out, error=f"No usable XND quotes for {day(expiration)}. If the market is closed and this expiry is new, try again after the open.")
    smile = core.smile_table(chain, F, T, rate)
    puts = core.puts_table(chain, F, T, rate, smile)
    vol = core.atm_vol(chain, F, T, rate)
    model = model_for(h, str(hist.index[-1].date()), hist)
    table, info = core.find_spreads(puts, target, cap, commission=commission, fill=fill, max_short_delta=max_delta, max_rel_spread=max_gap)
    if len(table):
        ca = [core.chance_above(model, F, k) for k in table["short"]]
        table["model_p"], table["model_lo"], table["model_hi"] = [c["p"] for c in ca], [c["lo"] for c in ca], [c["hi"] for c in ca]
        table["model_value"] = [core.model_value(model, F, row, commission)["mean"] for row in table.to_dict("records")]
    tested = core.tested_spread(puts, cap, commission=commission, fill=fill, max_rel_spread=max_gap)
    value = core.model_value(model, F, tested, commission) if tested else None
    out.update(chain=chain, T=T, h=h, F=F, smile=smile, puts=puts, vol=vol, implied=core.implied_range(smile, F, T, vol), model=model,
               mrange=core.model_range(model), table=table, info=info, tested=tested, value=value,
               tested_chance=core.chance_above(model, F, tested["short"]) if tested else None,
               rec=core.recommend(horizon, trend, tested, value, out["days_left"]))
    return out


with st.spinner("Reading the option quotes"):
    week = analyse("week", week_exp, float(week_target), account * week_cap_pct / 100)
    month = analyse("month", month_exp, float(month_target), account * month_cap_pct / 100)

# ---------------------------------------------------------------- top line
first_ok = next((x for x in (week, month) if "error" not in x), None)
c1, c2, c3, c4 = st.columns(4)
if first_ok:
    metric(c1, "XND (Nasdaq-100 ÷ 100)", f"{first_ok['F']:,.2f}", f"QQQ ≈ {usd(first_ok['F'] * qqq_per_point)}",
              help="The index level the options are priced around. One XND option covers 100 times this, about a seventh of a QQQ-sized Nasdaq-100 option.")
else:
    metric(c1, "XND (Nasdaq-100 ÷ 100)", "no quotes")
metric(c2, "VXN, last close", f"{hist['vxn'].iloc[-1]:.1f}", f"closes through {day(hist.index[-1])}",
          help="The market's volatility index for the Nasdaq-100: the yearly move in percent that option prices imply.")
metric(c3, "Index vs 200-day average", "Above" if trend["above"] else "Below",
          f"by {pct(abs(trend['gap']))} · rule says {'trade' if trend['above'] else 'stand aside'}",
          help="The only rule that passed the 2020-26 backtests: open a monthly spread only when the index closed above its average of the last 200 days.")
if first_ok:
    label = {"demo": "made-up quotes", "live": "live", "earlier today": "latest on record", "last close": "last session's close"}[first_ok["source"]]
    if first_ok["source"] == "live" and (now - first_ok["ts"]).total_seconds() > 20 * 60:
        label = "latest on record"                              # a snapshot taken while the market is closed
    metric(c4, "Option quotes", f"{first_ok['ts']:%-I:%M %p} ET", f"{day(first_ok['ts'])} · {label}")
else:
    metric(c4, "Option quotes", "none")

if demo:
    st.info("Demo quotes: the option prices below are made up from the real index level and VXN. Add your ThetaData key in the app's secrets and switch the demo off for real quotes.", icon=":material/science:")
if hist_source == "saved copy only" and (now.date() - hist.index[-1].date()).days > 4:
    st.warning(f"Couldn't refresh the index history, so the 200-day rule and the model use saved closes through {day(hist.index[-1])}.", icon=":material/history:")


def trade_line(s, expiration):
    n = int(s["n"])
    return f"Sell {n} × XND {s['short']:g} put, buy {n} × XND {s['long']:g} put, both expiring {day(expiration)}"


def show_rec(box, x, compact=True):
    rec = x["rec"]
    if rec["action"] == "trade":
        box.success(rec["headline"], icon=":material/check_circle:")
    else:
        box.warning(rec["headline"], icon=":material/pause_circle:")
    if compact and x["tested"] and x["tested"]["n"] >= 1:
        s = x["tested"]
        box.markdown(esc(f"**{trade_line(s, x['expiration'])}**  \nCollect about **{usd(s['premium'])}** · most you can lose **{usd(s['max_loss'])}**"))


st.subheader("The short answer")
left, right = st.columns(2)
for col, x, name in ((left, week, "This week"), (right, month, "This month")):
    box = col.container(border=True)
    box.markdown(f"**{name}** · expires {day(x['expiration'])}")
    if "error" in x:
        box.error(x["error"])
        continue
    show_rec(box, x)
    box.caption(f"Options imply a move of about ±{pct(x['implied']['move'])} by then. The model expects ±{pct(x['mrange']['move'])} "
                f"(usually between ±{pct(x['mrange']['move_low'])} and ±{pct(x['mrange']['move_high'])}).")


# ---------------------------------------------------------------- the range chart
def range_chart(x):
    blue, orange, ink, surface = ("#3987e5", "#d95926", "#e8e6df", "#0e1117") if dark_mode() else ("#2a78d6", "#eb6834", "#33322e", "#ffffff")
    F = x["F"]

    def span(lo, hi):
        return f"{signed(lo)} to {signed(hi)}"

    rows = []
    for name, r in (("What options imply", x["implied"]), ("What the model expects", x["mrange"])):
        rows.append(dict(row=name, q10=r["q10"], q25=r["q25"], q50=r["q50"], q75=r["q75"], q90=r["q90"],
                         wide=span(r["q10"], r["q90"]), half=span(r["q25"], r["q75"]),
                         xnd=f"{F * (1 + r['q10']):,.1f} to {F * (1 + r['q90']):,.1f}",
                         qqq=f"{usd(F * (1 + r['q10']) * qqq_per_point)} to {usd(F * (1 + r['q90']) * qqq_per_point)}"))
    bands = pd.DataFrame(rows)
    marks = []
    if len(x["table"]):
        marks.append(dict(row="Strikes you would sell", what=f"best chance {x['table'].iloc[0]['short']:g}", strike=float(x["table"].iloc[0]["short"])))
    if x["tested"]:
        marks.append(dict(row="Strikes you would sell", what=f"tested {x['tested']['short']:g}", strike=float(x["tested"]["short"])))
    marks = pd.DataFrame(marks)
    if len(marks):
        if marks["strike"].nunique() == 1:
            marks = pd.DataFrame([dict(row="Strikes you would sell", what=f"best chance and tested {marks['strike'].iloc[0]:g}", strike=marks["strike"].iloc[0])])
        marks["move"] = marks["strike"] / F - 1
        marks["detail"] = [f"{k:g}, {pct(abs(m))} below the index (QQQ ≈ {usd(k * qqq_per_point)})" for k, m in zip(marks["strike"], marks["move"])]
        marks = marks.sort_values("move").reset_index(drop=True)
        marks["dy"] = [(-14 if i % 2 == 0 else 16) for i in range(len(marks))]
    order = ["What options imply", "What the model expects"] + (["Strikes you would sell"] if len(marks) else [])
    y = alt.Y("row:N", sort=order, title=None, scale=alt.Scale(domain=order), axis=alt.Axis(labelLimit=220, ticks=False, domain=False, labelPadding=10))
    colour = alt.Color("row:N", scale=alt.Scale(domain=order[:2], range=[blue, orange]), legend=None)
    tips = [alt.Tooltip("row:N", title=" "), alt.Tooltip("half:N", title="Half the time"), alt.Tooltip("wide:N", title="8 times in 10"),
            alt.Tooltip("xnd:N", title="…as XND"), alt.Tooltip("qqq:N", title="…as QQQ")]
    lo = min(float(bands["q10"].min()), float(marks["move"].min()) if len(marks) else 0.0)
    hi = float(bands["q90"].max())
    pad = 0.12 * (hi - lo)
    xs = alt.Scale(domain=[lo - pad, hi + pad], nice=False)
    base = alt.Chart(bands)
    layers = [
        alt.Chart(pd.DataFrame({"v": [0.0]})).mark_rule(strokeDash=[3, 3], color=ink, opacity=0.45).encode(x=alt.X("v:Q", scale=xs)),
        base.mark_rule(strokeWidth=3, strokeCap="round").encode(x=alt.X("q10:Q", scale=xs, title="Change in the index by expiry", axis=alt.Axis(format="+.0%", tickCount=8)), x2="q90:Q", y=y, color=colour),
        base.mark_bar(height=16, cornerRadius=4).encode(x=alt.X("q25:Q", scale=xs), x2="q75:Q", y=y, color=colour),
        base.mark_tick(thickness=2, size=16, color=surface).encode(x=alt.X("q50:Q", scale=xs), y=y),
        base.mark_bar(height=34, opacity=0).encode(x=alt.X("q10:Q", scale=xs), x2="q90:Q", y=y, tooltip=tips),
    ]
    if len(marks):
        m = alt.Chart(marks)
        mt = [alt.Tooltip("what:N", title="Strike"), alt.Tooltip("detail:N", title=" ")]
        layers.append(m.mark_point(shape="diamond", size=110, filled=True, color=ink, stroke=surface, strokeWidth=2, opacity=1).encode(x=alt.X("move:Q", scale=xs), y=y))
        layers.append(m.mark_point(size=900, filled=True, opacity=0).encode(x=alt.X("move:Q", scale=xs), y=y, tooltip=mt))      # a bigger target to hover
        for dy in sorted(set(marks["dy"])):
            layers.append(alt.Chart(marks[marks["dy"] == dy]).mark_text(dy=dy, fontSize=12, color=ink).encode(x=alt.X("move:Q", scale=xs), y=y, text="what:N"))
    chart = alt.layer(*layers).properties(height=64 * len(order) + 30).configure_view(strokeWidth=0).configure_axis(grid=False)
    return chart, bands


BACKTESTS = pd.DataFrame([
    ["Hold QQQ, never sell", "2020-26", "+17.3%", "−29.8%", "The yardstick."],
    ["25-day put spread every month", "2020-26", "+17.8%", "−35.8%", "Barely ahead, with a deeper fall."],
    ["25-day put spread, only above the 200-day average", "2020-26", "+19.4%", "−31.8%", "The only rule that passed. Its edge is three skipped months."],
    ["Same rule, estimated prices", "2001-19", "+8.3% vs +9.0% for QQQ", "", "No edge before 2020."],
    ["Weekly put spreads", "2020-26", "+16.1%", "−32.8%", "Lost to QQQ. Costs take about 18% of each premium."],
    ["Weekly put spreads, estimated prices", "2001-19", "lost in every era", "", "Average result below zero per dollar at risk."],
    ["A model picking which spread to hold", "2022-26", "+5.4% vs +11.6% for QQQ", "", "It chased whatever had just worked."],
], columns=["What was tested", "Years", "After-tax return per year", "Worst fall", "What it showed"])


def detail(x, name):
    if "error" in x:
        st.error(x["error"])
        return
    F, exp, imp, mr, chk = x["F"], x["expiration"], x["implied"], x["mrange"], x["model"]["check"]
    st.caption(f"Expires {day(exp)} · {x['days_left']} days, {x['h']} trading session{'s' if x['h'] != 1 else ''} · quotes from {x['ts']:%-I:%M %p} ET, {day(x['ts'])}")

    # 1. the move
    st.markdown(f"#### How far could the index move by {day(exp)}?")
    a, b, c = st.columns(3)
    metric(a, "Options imply", f"±{pct(imp['move'])}", f"about ±{usd(F * imp['move'] * qqq_per_point)} on QQQ",
             help="The one-standard-deviation move priced into at-the-money XND options. QQQ tracks the same index, so its implied move in percent is the same.")
    metric(b, "The model expects", f"±{pct(mr['move'])}", f"usually ±{pct(mr['move_low'])} to ±{pct(mr['move_high'])}",
             help="The typical move the model predicts from VXN and how much the index really moved over the last 5, 20 and 60 days. "
                  "The second line is its error bar: in 8 of 10 past cases the actual size of the moves landed in that range.")
    gap = imp["move"] / mr["move"] - 1
    metric(c, "Options against the model", signed(gap, 0), f"{'more' if gap >= 0 else 'less'} movement priced in",
             help=f"Options normally price in more movement than arrives: VXN has run {pct(chk['vxn_overstates'], 0)} above what followed, on average, over these horizons. "
                  "That gap is the premium a put spread collects. Costs and the occasional large fall are what eat it.")
    chart, bands = range_chart(x)
    st.altair_chart(chart, width="stretch")
    st.caption("Thick bar: where the index finishes half the time. Thin line: 8 times in 10. The notch in the bar is the middle. "
               "The model says how big the move is likely to be. It does not say which way: nothing tested could predict direction.")
    with st.expander("Numbers behind the chart"):
        t = bands[["row", "q10", "q25", "q50", "q75", "q90"]].copy()
        for q in ("q10", "q25", "q50", "q75", "q90"):
            t[q] = [f"{signed(v)}  ({F * (1 + v):,.1f})" for v in t[q]]
        t.columns = ["", "1 in 10 finish below", "1 in 4 below", "Middle", "1 in 4 above", "1 in 10 above"]
        st.dataframe(t, hide_index=True, width="stretch")
        if not imp["from_smile"]:
            st.caption("The quotes didn't reach far enough to read the options' range strike by strike, so it is drawn from at-the-money volatility alone.")

    # 2. best chance for the target
    st.markdown(esc(f"#### Best chance to collect {usd(x['target'])}"))
    table = x["table"]
    if not len(table):
        most = x["info"]["most_within_cap"]
        if most:
            st.warning(esc(f"No spread collects {usd(x['target'])} while keeping the worst case under {usd(x['cap'])}. The most that cap allows is about "
                       f"**{usd(most['premium'])}** (sell {most['n']} × {most['short']:g} put, buy {most['n']} × {most['long']:g} put). "
                       "Lower the target or raise the loss cap in the sidebar."), icon=":material/block:")
        else:
            st.warning(esc(f"No spread fits a loss cap of {usd(x['cap'])} with these quotes. Raise the cap or loosen the filters in the sidebar."), icon=":material/block:")
    else:
        best = table.iloc[0]
        st.markdown(f"**{trade_line(best, exp)}**")
        a, b, c, d = st.columns(4)
        metric(a, "You collect", usd(best["premium"]), f"after {usd(2 * best['n'] * commission)} commission")
        metric(b, "Most you can lose", usd(best["max_loss"]), f"{pct(best['max_loss'] / account)} of the account")
        metric(c, "Strike you sell", f"{best['short']:g}", f"{pct(best['below'])} below · QQQ ≈ {usd(best['short'] * qqq_per_point)}")
        metric(d, "Chance you keep it all", pct(best["model_p"], 0), f"model · range {pct(best['model_lo'], 0)} to {pct(best['model_hi'], 0)}",
               help="The chance the index finishes above the strike you sell. It is how often that happened across 20 years of comparable moves, "
                    "with a range for how sure that count is. What the option prices imply is in the table below.")
        safest_loss = table.loc[table["max_loss"].idxmin()]
        if safest_loss["short"] != best["short"]:
            st.caption(esc(f"This is the spread furthest from the index that still reaches {usd(x['target'])} inside your cap, so it has the best chance of paying in full "
                           f"(options put it at {pct(best['p_above'], 0)}). It also uses the most of your cap. The same premium with the least at risk: "
                           f"sell {int(safest_loss['n'])} × {safest_loss['short']:g}, buy {int(safest_loss['n'])} × {safest_loss['long']:g}, "
                           f"most you can lose {usd(safest_loss['max_loss'])}, chance {pct(safest_loss['model_p'], 0)}."))
        else:
            st.caption(f"Options put the chance at {pct(best['p_above'], 0)}.")
        show = pd.DataFrame({
            "Sell / buy": [f"{k:g} / {l:g}" for k, l in zip(table["short"], table["long"])], "Qty": table["n"], "Collect": table["premium"],
            "Max loss": table["max_loss"], "Below index": table["below"] * 100, "QQQ level": table["short"] * qqq_per_point,
            "Chance (model)": [f"{p:.0%}  ({lo:.0%} to {hi:.0%})" for p, lo, hi in zip(table["model_p"], table["model_lo"], table["model_hi"])],
            "Chance (options)": table["p_above"] * 100, "Model value": table["model_value"] * 100})
        styled = show.style.format({"Collect": "${:,.0f}", "Max loss": "${:,.0f}", "Below index": "{:.1f}%", "QQQ level": "${:,.0f}",
                                    "Chance (options)": "{:.0f}%", "Model value": "{:+.1f}¢"})
        st.dataframe(styled, hide_index=True, width="stretch", column_config={
            "Sell / buy": st.column_config.TextColumn(width="small", help="Strike of the put you sell, then the put you buy to limit the loss."),
            "Qty": st.column_config.NumberColumn(width="small", help="How many spreads."),
            "Collect": st.column_config.NumberColumn(width="small", help="Premium after commission."),
            "Max loss": st.column_config.NumberColumn(width="small", help="What you lose if the index finishes at or below the strike you buy."),
            "Below index": st.column_config.NumberColumn(width="small", help="How far the strike you sell sits below the index."),
            "QQQ level": st.column_config.NumberColumn(width="small", help="Where QQQ would be with the index at the strike you sell."),
            "Chance (model)": st.column_config.TextColumn(help="Chance the index finishes above the strike you sell, by the model, with a range for how sure that count is (8 in 10)."),
            "Chance (options)": st.column_config.NumberColumn(help="The same chance, as option prices imply it."),
            "Model value": st.column_config.NumberColumn(width="small", help=esc("Per $1 at risk: what this spread would have made on average across the model's history of comparable moves, after costs. Around zero is normal."))})
        export = pd.DataFrame({
            "expiration": str(exp), "sell_put_strike": table["short"], "buy_put_strike": table["long"], "spreads": table["n"],
            "credit_per_spread": table["credit"].round(2), "premium_after_commission": table["premium"].round(2), "max_loss": table["max_loss"].round(2),
            "breakeven": table["breakeven"].round(2), "sold_strike_below_index": table["below"].round(4), "sold_strike_delta": table["delta"].round(3),
            "chance_above_model": table["model_p"].round(3), "chance_above_model_low": table["model_lo"].round(3), "chance_above_model_high": table["model_hi"].round(3),
            "chance_above_options": table["p_above"].round(3), "model_value_per_dollar_at_risk": table["model_value"].round(4)})
        st.caption("Every row reaches the target inside your loss cap, safest first. For each strike sold, the strike bought is the one that leaves the least at risk. "
                   + ("Prices assume you sell at the bid and buy at the ask; a limit order between the two often does better." if fill == "quote"
                      else "Prices assume a fill halfway between bid and ask, which a limit order may not get."))
        st.download_button("Download these spreads (CSV)", export.to_csv(index=False), file_name=f"xnd_spreads_{name}_{exp}.csv", mime="text/csv", key=f"dl_{name}")

    # 3. the tested trade
    st.markdown("#### The trade the backtests used, and whether to take it")
    show_rec(st, x, compact=False)
    for n in x["rec"]["notes"]:
        st.markdown(f"- {n}")
    s, v, ch = x["tested"], x["value"], x["tested_chance"]
    if s and s["n"] >= 1:
        st.markdown(f"**{trade_line(s, exp)}**")
        a, b, c, d = st.columns(4)
        metric(a, "You collect", usd(s["premium"]), f"{usd(s['net_each'])} a spread, after fees")
        metric(b, "Most you can lose", usd(s["max_loss"]), f"{pct(s['max_loss'] / account)} of the account")
        metric(c, "Strike you sell", f"{s['short']:g}", f"{pct(s['below'])} below · QQQ ≈ {usd(s['short'] * qqq_per_point)}",
                 help=f"The put closest to 20 delta ({abs(s['delta']):.2f} here). The put bought is the one closest to 10 delta ({abs(s['long_delta']):.2f}).")
        metric(d, "Model's value", cents(v["mean"]), f"range {cents(v['lo'])} to {cents(v['hi'])}",
               help="Per dollar at risk: what this spread would have made on average across the model's history of comparable moves, after costs, with a range for how sure that average is.")
        st.caption(f"By the model this spread makes {cents(v['mean'])} per dollar at risk on average: it pays in {pct(v['win'], 0)} of cases and loses the maximum in {pct(v['full_loss'], 0)}. "
                   f"Chance the index finishes above {s['short']:g}: {pct(ch['p'], 0)} by the model (range {pct(ch['lo'], 0)} to {pct(ch['hi'], 0)}), {pct(s['p_above'], 0)} by the options. "
                   f"You break even at {s['breakeven']:,.2f}.")
    elif s:
        st.caption(esc(f"One of these spreads risks {usd(s['risk_each'])}, more than your loss cap of {usd(x['cap'])}."))


st.subheader("The detail")
tab_week, tab_month = st.tabs(["This week", "This month"])
with tab_week:
    detail(week, "week")
with tab_month:
    detail(month, "month")

# ---------------------------------------------------------------- background
st.subheader("Behind the numbers")
with st.expander("How good is the model?"):
    st.markdown("The model predicts **how big** the move will be, from VXN and from how much the index really moved over the last 5, 20 and 60 days. "
                "It then looks at 20 years of past moves, each scaled by what the model predicted at the time, to draw the range. "
                "Every year below was predicted using only earlier years, so this is how it did on days it had not seen.")
    cols = {}
    for x, name in ((week, "This week"), (month, "This month")):
        if "error" in x:
            continue
        k = x["model"]["check"]
        cols[f"{name} ({x['h']} sessions)"] = {
            "Years checked": k["years"], "Finished inside the model's 8-in-10 range": pct(k["inside"], 0), "Finished below it": pct(k["below"], 0),
            "Finished inside a range drawn from VXN alone": pct(k["inside_vxn"], 0),
            "Typical error in the size of the move: model": pct(k["error_model"], 0), "Typical error in the size of the move: VXN alone": pct(k["error_vxn"], 0)}
    if cols:
        st.table(pd.DataFrame(cols))
    st.markdown("- A good 8-in-10 range holds about 80% of the time. A range from VXN alone holds closer to 90%: it is too wide, because options price in more movement than arrives.\n"
                "- The model's size estimate is still off by a quarter to a third on average. That is the error bar shown next to its expected move.\n"
                "- Inputs right now: " + (", ".join(f"{lab} {first_ok['model']['inputs'][key]:.0%}" for key, lab in
                                                     (("vxn", "VXN"), ("rv5", "5-day movement"), ("rv20", "20-day"), ("rv60", "60-day"))) if first_ok else "none")
                + f" (yearly rates, closes through {day(hist.index[-1])}).")
with st.expander("What the backtests found"):
    st.table(BACKTESTS.set_index("What was tested"))
    st.markdown("Returns are after tax on a \\$250,000 account that holds QQQ and never sells it, with XND spreads on top (6% of the account at risk per monthly spread). "
                "Nothing tested beat simply holding QQQ in a way that held up across 25 years. Treat the monthly rule as a small bet with thin evidence, and the weekly spread as one to skip.")
with st.expander("Data check"):
    cols = {}
    for x, name in ((week, "This week"), (month, "This month")):
        if "error" in x:
            cols[name] = {"Expiry": str(x["expiration"]), "Quotes": x["error"]}
            continue
        cols[name] = {"Expiry": str(x["expiration"]), "Quotes": x["source"], "Quote time (ET)": f"{x['ts']:%Y-%m-%d %H:%M}",
                      "Strikes with quotes": str(int(x["chain"]["strike"].nunique())), "Index level from options": f"{x['F']:,.2f}",
                      "At-the-money volatility": pct(x["vol"]), "Puts the finder looked at": str(x["info"]["shorts_considered"]),
                      "Notes": "; ".join(f"{k}: {v}" for k, v in x["notes"].items())}
    st.table(pd.DataFrame(cols).fillna(""))
    st.caption(esc(f"Index history: {len(hist):,} daily closes, {hist.index[0]:%Y} to {day(hist.index[-1])} {hist.index[-1]:%Y} ({hist_source}). "
               f"200-day average {trend['average']:,.0f}, last close {trend['last']:,.0f}. T-bill rate used: {pct(rate, 2)}. "
               f"One XND point ≈ {usd(qqq_per_point, cents=True)} on QQQ."))
    for x, name in ((week, "week"), (month, "month")):
        if "error" not in x:
            st.download_button(f"Download the {name}'s option quotes (CSV)", x["chain"].to_csv(index=False), file_name=f"xnd_quotes_{x['expiration']}.csv",
                               mime="text/csv", key=f"chain_{name}")
