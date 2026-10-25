"""SkyRecover web dashboard.  Run locally with:  streamlit run app.py"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))  # so the app works without `pip install -e .` (e.g. on Streamlit Cloud)

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from skyrecover.optimize import build_scenario, compare, optimality_gap  # noqa: E402
from skyrecover.ripple import add_rotations, propagate  # noqa: E402

st.set_page_config(page_title="SkyRecover", page_icon="✈️", layout="wide")

DEMO = ROOT / "data" / "demo"


# ---------- colors (validated chart palette, light + dark steps) ----------
def colors():
    dark = getattr(st.context, "theme", None) is not None and st.context.theme.type == "dark"
    return {
        "blue": "#3987e5" if dark else "#2a78d6",
        "orange": "#d95926" if dark else "#eb6834",
        "aqua": "#199e70" if dark else "#1baf7a",
    }


OPSDESK = {
    "font": "Barlow, system-ui, sans-serif", "axis": {"labelFont": "Barlow, system-ui, sans-serif", "titleFont": "Barlow, system-ui, sans-serif", "labelColor": "#4A5866",
                               "titleColor": "#4A5866", "gridColor": "#D5DDE2", "domainColor": "#B9C4CB"},
    "legend": {"labelFont": "Barlow, system-ui, sans-serif", "labelColor": "#14212E"},
    "title": {"font": "B612, Barlow, system-ui, sans-serif", "color": "#14212E", "subtitleFont": "Barlow, system-ui, sans-serif", "subtitleColor": "#4A5866"},
    "view": {"stroke": None}, "background": "transparent"}


def styled(chart):
    """Apply the ops-desk chart style (fonts, quiet axes) to any Altair chart."""
    return chart.configure(**{k: v for k, v in OPSDESK.items() if k not in ("axis", "legend", "title", "view")}) \
        .configure_axis(**OPSDESK["axis"]).configure_legend(**OPSDESK["legend"]) \
        .configure_title(**OPSDESK["title"]).configure_view(**OPSDESK["view"])

# ---------- data ----------
@st.cache_data
def load():
    flights = pd.read_parquet(DEMO / "flights.parquet")
    preds = pd.read_parquet(DEMO / "predictions.parquet")
    metrics = json.loads((DEMO / "metrics.json").read_text())
    return flights, preds, metrics


@st.cache_data
def rotations(flights):
    return add_rotations(flights)


@st.cache_data(show_spinner=False)
def run_recovery(date, hub, start, hours, mode, pool, time_limit):
    flights, _, _ = load()
    kw = {"aog_tail": "auto"} if mode == "aog" else {"ground_delay": 60}
    problem, info = build_scenario(flights, date, hub, start, hours=hours, pool=pool, **kw)
    plans = compare(problem, time_limit=time_limit)
    out = {}
    for k, p in plans.items():
        out[k] = {"table": p.table if p.rows else None,
                  "summary": p.summary() if p.rows else None,
                  "gap": optimality_gap(p), "status": p.status}
    ready = {a.tail: a.ready for a in problem.aircraft}
    return out, info, ready


def hhmm(m):
    m = int(m)
    return f"{m // 60 % 24:02d}:{m % 60:02d}"


flights, preds, metrics = load()
C = colors()
AIRPORTS = pd.read_csv(DEMO / "airports.csv").set_index("code")

# ---------- look and feel: "ops desk" (aeronautical chart paper + flight strips) ----------
st.html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=B612:wght@400;700&family=Barlow:wght@400;500;600&display=swap');
:root {
  --paper: #EDF1F2; --ink: #14212E; --ink-soft: #4A5866; --magenta: #B0246E;
  --chart-blue: #1F5AA6; --buff: #EFE3C2; --buff-edge: #CDBB8E; --rule: #B9C4CB;
}
html, body, .stApp, [data-testid="stMarkdownContainer"], .stTabs, input, textarea, button, label,
[data-baseweb="select"], [data-testid="stWidgetLabel"] { font-family: 'Barlow', system-ui, sans-serif; }
h1, h2, h3, h4, [data-testid="stMetricValue"], .stTabs [role="tab"] p { font-family: 'B612', 'Barlow', sans-serif !important; letter-spacing: 0; }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.2rem; max-width: 1180px; }
h2 { font-size: 1.45rem !important; font-weight: 700 !important; margin-top: .4rem !important; }
.stTabs [role="tablist"] { gap: 1.6rem; border-bottom: 1px solid var(--rule); }
.stTabs [role="tab"] p { font-size: 1rem; }
a { color: var(--chart-blue); }
:focus-visible { outline: 2px solid var(--magenta) !important; outline-offset: 2px; }

/* masthead */
.sr-mast { display:flex; align-items:baseline; justify-content:space-between; gap:1rem;
  border-bottom: 2px solid var(--ink); padding-bottom: .55rem; margin-bottom: 2.2rem; }
.sr-mark { font-family:'B612',sans-serif; font-weight:700; font-size:1.35rem; color:var(--ink); }
.sr-mark span { color: var(--magenta); }
.sr-mast a { font-size: .95rem; text-decoration: none; }

/* hero */
.sr-hero h1 { font-family:'B612',sans-serif !important; font-weight:700; color:var(--ink);
  font-size: clamp(1.9rem, 4.2vw, 3.1rem); line-height:1.12; max-width: 24ch; text-wrap: balance; margin: 0 0 .9rem 0; padding:0; }
.sr-hero p.lede { font-size:1.12rem; line-height:1.55; color:var(--ink-soft); max-width: 62ch; margin:0 0 1.8rem 0; }
.sr-plane { font-family:'B612',sans-serif; font-size:.88rem; color:var(--ink-soft); margin-bottom:.6rem; }

/* flight strips: buff paper, the way departure strips are printed */
.sr-strips { display:flex; gap:.75rem; flex-wrap:wrap; position:relative; padding-bottom:.4rem; }
.sr-strip { flex: 1 1 165px; background:var(--buff); border:1px solid var(--buff-edge); border-radius:3px;
  display:grid; grid-template-columns: 1fr auto; grid-template-rows:auto auto auto;
  padding:.55rem .7rem .6rem .8rem; position:relative; color:var(--ink); min-width:150px; }
.sr-strip::before { content:""; position:absolute; left:0; top:0; bottom:0; width:5px; background:var(--chart-blue); border-radius:3px 0 0 3px; }
.sr-strip .fn { font-family:'B612',sans-serif; font-weight:700; font-size:.95rem; }
.sr-strip .tm { font-family:'B612',sans-serif; font-size:.85rem; color:var(--ink-soft); text-align:right; }
.sr-strip .rt { grid-column:1 / 3; font-family:'B612',sans-serif; font-size:1.35rem; font-weight:700; margin:.15rem 0 .25rem; }
.sr-strip .dl { grid-column:1 / 3; display:flex; align-items:center; gap:.45rem; font-size:.92rem; }
.sr-strip .dl b { font-family:'B612',sans-serif; color:var(--magenta); font-size:1.05rem; }
.sr-strip .bar { flex:1; height:6px; background:#D8C9A0; border-radius:3px; overflow:hidden; }
.sr-strip .bar i { display:block; height:100%; background:var(--magenta); border-radius:3px; width:var(--w); }
.sr-strip.origin::before { background: var(--magenta); }

/* the one moment of motion: the delay travels strip to strip on load */
@media (prefers-reduced-motion: no-preference) {
  .sr-strip .dl { opacity:0; animation: sr-arrive .5s ease-out forwards; animation-delay: var(--d); }
  .sr-strip .bar i { width:0; animation: sr-fill .7s ease-out forwards; animation-delay: var(--d); }
}
@keyframes sr-arrive { to { opacity:1; } }
@keyframes sr-fill { to { width: var(--w); } }
.sr-caption { font-size:.9rem; color:var(--ink-soft); margin-top:.7rem; max-width:70ch; }

/* findings: three sentences, not three dashboard tiles */
.sr-findings { display:grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1.6rem;
  margin: 2.4rem 0 2.6rem; border-top:1px solid var(--rule); padding-top:1.4rem; }
.sr-findings div { font-size:1rem; line-height:1.5; color:var(--ink); }
.sr-findings b { font-family:'B612',sans-serif; display:block; font-size:1.6rem; color:var(--chart-blue); margin-bottom:.2rem; }
.sr-foot { margin-top:3rem; border-top:1px solid var(--rule); padding-top:.8rem; font-size:.85rem; color:var(--ink-soft); }
</style>
""")

# ---------- hero: a real plane's real day (N349NW, Jan 23 2024) ----------
HERO_TAIL, HERO_DATE = "N349NW", "2024-01-23"
hero = flights[(flights["tail_number"] == HERO_TAIL) & (flights["date"] == pd.Timestamp(HERO_DATE))].sort_values("sched_dep")
max_d = max(1.0, hero["dep_delay"].max())
strips = []
for i, r in enumerate(hero.itertuples()):
    cls = "sr-strip origin" if i == 0 else "sr-strip"
    why = "first delay" if i == 0 else "late aircraft"
    strips.append(
        f'<div class="{cls}" style="--d:{0.35 + i * 0.55:.2f}s; --w:{r.dep_delay / max_d * 100:.0f}%">'
        f'<span class="fn">DL{r.flight_no}</span><span class="tm">{r.sched_dep:%H:%M}</span>'
        f'<span class="rt">{r.origin} &rsaquo; {r.dest}</span>'
        f'<span class="dl"><b>+{r.dep_delay:.0f}</b><span class="bar"><i></i></span>{why}</span></div>')
first, last = hero.iloc[0], hero.iloc[-1]


def clock12(ts):
    """9:15 am style, without %-I (which crashes on Windows)."""
    h = ts.hour % 12 or 12
    return f"{h}:{ts.minute:02d} {'am' if ts.hour < 12 else 'pm'}"


def long_date(ts):
    return f"{ts:%A %B} {ts.day}, {ts.year}"
rv = metrics["ripple_validation"]
day_of = metrics["models"]["day_of"]
ripple_only = metrics["models"]["ripple_formula"]

st.html(f"""
<div class="sr-mast">
  <div class="sr-mark">Sky<span>Recover</span></div>
  <a href="https://github.com/DaKSh32-blip/SkyRecover" target="_blank">Code and write-up on GitHub</a>
</div>
<section class="sr-hero">
  <h1>A {first.dep_delay:.0f}-minute delay at {clock12(first.sched_dep)} was still costing Delta at {clock12(last.sched_dep)}.</h1>
  <p class="lede">Most airline delays are old delays passing forward. A plane lands late, its next flight leaves late,
  and so on all day. SkyRecover follows every aircraft, predicts which flights are about to be hit, and re-plans
  the day when something breaks.</p>
  <div class="sr-plane">Aircraft {HERO_TAIL}, {long_date(pd.Timestamp(HERO_DATE))}, departure delays in minutes</div>
  <div class="sr-strips">{''.join(strips)}</div>
  <p class="sr-caption">Real flights from US DOT records. Delta reported every delay after the first as
  "late aircraft": the plane was still catching up.</p>
</section>
<div class="sr-findings">
  <div><b>{rv['correlation']:.2f}</b>correlation between SkyRecover's ripple estimate and the late-aircraft delay Delta reported, across {rv['flights_compared']:,} delayed flights.</div>
  <div><b>{day_of['recall']:.0%}</b>of late departures flagged an hour ahead on a month the model never saw, {day_of['recall'] / ripple_only['recall']:.1f}x more than slack math alone.</div>
  <div><b>0 cancelled</b>when a plane breaks down at Atlanta for five hours. Without aircraft swaps, the best plan cancels a round trip. Both plans are proven optimal.</div>
</div>
""")

tab_net, tab_day, tab_ripple, tab_recover, tab_model = st.tabs(
    ["Network", "Airport day", "Ripple", "Recovery", "How well it predicts"])

# =====================================================================
# NETWORK MAP
# =====================================================================
with tab_net:
    import pydeck as pdk

    st.subheader("Where the delays go")
    st.caption("Every Delta route out of one airport. Arc color shows average departure delay, thickness shows how many flights.")
    hubs_all = flights["origin"].value_counts().index.tolist()
    n1, n2 = st.columns([1, 2])
    nhub = n1.selectbox("Airport", hubs_all, index=hubs_all.index("ATL"), key="net_hub")
    months = {"January": 1, "February": 2, "March": 3}
    nmonth = n2.radio("Month", list(months), horizontal=True, index=2, key="net_month")
    sub = flights[(flights["origin"] == nhub) & (flights["sched_dep"].dt.month == months[nmonth]) & (~flights["cancelled"])]
    routes = sub.groupby("dest").agg(flights=("flight_id", "count"),
                                     avg_delay=("dep_delay", lambda x: x.clip(lower=0).mean()),
                                     late_share=("dep_delay", lambda x: (x >= 15).mean())).reset_index()
    routes = routes[routes["dest"].isin(AIRPORTS.index)]
    o = AIRPORTS.loc[nhub]
    routes["src_lon"], routes["src_lat"] = o["lon"], o["lat"]
    routes["dst_lon"] = routes["dest"].map(AIRPORTS["lon"])
    routes["dst_lat"] = routes["dest"].map(AIRPORTS["lat"])
    routes["city"] = routes["dest"].map(AIRPORTS["city"])

    # sequential single-hue ramp (light blue = small delay, dark blue = big delay)
    ramp = ["#b7d3f6", "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
    edges = [0, 6, 10, 14, 18, 24, 10**6]
    def ramp_rgb(v):
        k = next(i for i in range(len(ramp)) if v < edges[i + 1])
        h = ramp[k].lstrip("#")
        return [int(h[j:j + 2], 16) for j in (0, 2, 4)] + [230]
    routes["color"] = routes["avg_delay"].map(ramp_rgb)
    routes["width"] = 1.5 + 6 * (routes["flights"] / routes["flights"].max()) ** 0.5
    routes["delay_txt"] = routes["avg_delay"].round(0).astype(int).astype(str) + " min"
    routes["late_txt"] = (routes["late_share"] * 100).round(0).astype(int).astype(str) + "%"

    arcs = pdk.Layer("ArcLayer", routes, get_source_position=["src_lon", "src_lat"],
                     get_target_position=["dst_lon", "dst_lat"], get_source_color="color",
                     get_target_color="color", get_width="width", pickable=True, auto_highlight=True,
                     great_circle=True)
    dots = pdk.Layer("ScatterplotLayer", routes, get_position=["dst_lon", "dst_lat"], get_radius=22000,
                     get_fill_color=[20, 33, 46, 200], pickable=True)
    view = pdk.ViewState(latitude=37.5, longitude=-96, zoom=3.1, pitch=30)
    st.pydeck_chart(pdk.Deck(layers=[arcs, dots], initial_view_state=view, map_provider="carto", map_style="light",
                             tooltip={"text": "{city} ({dest})\n{flights} flights, avg delay {delay_txt}, {late_txt} left 15+ min late"}),
                    height=520)
    legend = "".join(f'<span style="display:inline-flex;align-items:center;gap:.35rem;margin-right:1rem">'
                     f'<i style="display:inline-block;width:22px;height:8px;border-radius:4px;background:{c}"></i>{lab}</span>'
                     for c, lab in zip(ramp, ["under 6", "6 to 10", "10 to 14", "14 to 18", "18 to 24", "24+"]))
    st.html(f'<div style="font-size:.9rem;color:#4A5866">Average departure delay (min): {legend}</div>')

    worst = routes.sort_values("avg_delay", ascending=False).head(8)
    st.markdown(f"**Most delayed routes from {nhub} in {nmonth}**")
    st.dataframe(worst.assign(route=nhub + " to " + worst["dest"] + " (" + worst["city"] + ")")[
                 ["route", "flights", "avg_delay", "late_share"]].rename(columns={
                     "route": "Route", "flights": "Flights", "avg_delay": "Avg delay (min)", "late_share": "Left 15+ min late"}),
                 hide_index=True, width="stretch",
                 column_config={"Avg delay (min)": st.column_config.NumberColumn(format="%.0f"),
                                "Left 15+ min late": st.column_config.NumberColumn(format="percent")})

# =====================================================================
# AIRPORT DAY
# =====================================================================
with tab_day:
    st.subheader("A day at one airport")
    c1, c2 = st.columns(2)
    day = c1.date_input("Date", pd.Timestamp("2024-03-15"), min_value=flights["date"].min(),
                        max_value=flights["date"].max(), key="day_date")
    hubs = flights["origin"].value_counts().index.tolist()
    hub = c2.selectbox("Airport", hubs, index=hubs.index("ATL"), key="day_hub")

    dd = flights[(flights["date"] == pd.Timestamp(day)) & (flights["origin"] == hub) & (~flights["cancelled"])].copy()
    dd = dd.merge(preds, on="flight_id", how="left")
    has_pred = dd["prob_late"].notna().any()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Departures", f"{len(dd):,}")
    m2.metric("Left 15+ min late", f"{(dd['dep_delay'] >= 15).mean():.0%}" if len(dd) else "-")
    m3.metric("Avg departure delay", f"{dd['dep_delay'].clip(lower=0).mean():.0f} min" if len(dd) else "-")
    cancelled = flights[(flights["date"] == pd.Timestamp(day)) & (flights["origin"] == hub) & flights["cancelled"]]
    m4.metric("Cancelled", f"{len(cancelled):,}")

    if len(dd):
        dd["hour"] = dd["sched_dep"].dt.hour
        hourly = dd.groupby("hour").agg(avg_delay=("dep_delay", lambda s: s.clip(lower=0).mean()),
                                        flights=("flight_id", "count")).reset_index()
        st.markdown("**Delays build up through the day**")
        chart = alt.Chart(hourly).mark_bar(color=C["blue"], cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            x=alt.X("hour:O", title="Scheduled departure hour (local)", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("avg_delay:Q", title="Average departure delay (min)"),
            tooltip=[alt.Tooltip("hour:O", title="Hour"), alt.Tooltip("avg_delay:Q", title="Avg delay (min)", format=".0f"),
                     alt.Tooltip("flights:Q", title="Departures")],
        ).properties(height=260)
        st.altair_chart(styled(chart), width="stretch", theme=None)

        st.markdown("**Flights the model flagged an hour ahead**" if has_pred else "**Departures**")
        if has_pred:
            flagged = dd["prob_late"] >= 0.5
            late = dd["dep_delay"] >= 15
            caught = (flagged & late).sum()
            st.caption(f"Of {int(late.sum())} flights that left 15+ min late, the model flagged {int(caught)} "
                       f"an hour before departure. {int(flagged.sum())} flagged in total, "
                       f"{(caught / flagged.sum() if flagged.sum() else 0):.0%} of them really were late.")
        else:
            st.caption("Model predictions are shown for March only (the month it was tested on). "
                       "Pick a March date to see them.")
        show = dd.sort_values("prob_late" if has_pred else "dep_delay", ascending=False)
        show = show.assign(**{"Departs": show["sched_dep"].dt.strftime("%H:%M"),
                              "Flight": "DL" + show["flight_no"].astype(str)})
        cols = ["Departs", "Flight", "dest", "tail_number", "dep_delay"] + (["prob_late", "pred_delay"] if has_pred else [])
        st.dataframe(show[cols].rename(columns={"dest": "To", "tail_number": "Aircraft", "dep_delay": "Actual delay (min)",
                                                 "prob_late": "Predicted chance late", "pred_delay": "Predicted delay (min)"}),
                     hide_index=True, width="stretch", height=320,
                     column_config={"Predicted chance late": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
                                    "Predicted delay (min)": st.column_config.NumberColumn(format="%.0f")})

# =====================================================================
# RIPPLE
# =====================================================================
with tab_ripple:
    st.subheader("What if one flight lands late?")
    st.caption("A plane has some spare time on the ground between flights (slack). "
               "If it lands later than its slack can absorb, the next flight leaves late too, and so on.")
    rot = rotations(flights)
    c1, c2, c3 = st.columns([1, 1, 2])
    rday = c1.date_input("Date", pd.Timestamp("2024-03-23"), min_value=flights["date"].min(),
                         max_value=flights["date"].max(), key="rip_date")
    rhub = c2.selectbox("From airport", hubs, index=hubs.index("ATL"), key="rip_hub")
    cands = rot[(rot["date"] == pd.Timestamp(rday)) & (rot["origin"] == rhub) & (~rot["cancelled"])
                & (rot["next_flight_id"] >= 0)].sort_values("sched_dep")
    if cands.empty:
        st.info("No flights with a later connection from that airport on that day.")
    else:
        labels = {int(r.flight_id): f"{r.sched_dep:%H:%M}  DL{r.flight_no}  {r.origin}->{r.dest}  ({r.tail_number})"
                  for r in cands.itertuples()}
        fid = c3.selectbox("Flight", list(labels), format_func=labels.get, key="rip_flight")
        extra = st.slider("How late does it land? (minutes)", 0, 300, 90, 5)

        hit = propagate(rot, fid, extra)
        tail = rot.loc[rot["flight_id"] == fid, "tail_number"].iloc[0]
        plane = rot[(rot["tail_number"] == tail) & (rot["date"] == pd.Timestamp(rday))].sort_values("sched_dep").copy()
        added = dict(zip(hit["flight_id"], hit["added_delay"]))
        added[fid] = extra
        plane["added"] = plane["flight_id"].map(added).fillna(0)
        plane["label"] = plane.apply(lambda r: f"DL{r.flight_no} {r.origin}->{r.dest}", axis=1)

        if hit.empty:
            st.success(f"No ripple. The next flight's slack absorbs all {extra} minutes.")
        else:
            st.warning(f"**{len(hit)} later flights** on {tail} leave late, "
                       f"**{hit['added_delay'].sum():,.0f} delay minutes** in total.")

        # One row per flight: faded bar = schedule, solid orange bar = where it moves with the delay
        # Bar length = real flight duration. (Using the local landing time would squash
        # bars to nothing when a flight crosses into an earlier time zone.)
        block = pd.to_timedelta(plane["block_minutes"], unit="m")
        plane["start"] = plane["sched_dep"]
        plane["end"] = plane["sched_dep"] + block
        shift = pd.to_timedelta(plane["added"].where(plane["flight_id"] != fid, 0), unit="m")
        plane["new_start"] = plane["sched_dep"] + shift
        plane["new_end"] = plane["new_start"] + block + pd.to_timedelta(
            plane["added"].where(plane["flight_id"] == fid, 0), unit="m")
        plane["kind"] = "As scheduled"
        order = plane["label"].tolist()
        y = alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=220))
        tips = [alt.Tooltip("label:N", title="Flight"),
                alt.Tooltip("sched_dep:T", title="Scheduled", format="%H:%M"),
                alt.Tooltip("added:Q", title="Minutes added"),
                alt.Tooltip("slack_minutes:Q", title="Slack before (min)")]
        planned = alt.Chart(plane).mark_bar(cornerRadius=4, opacity=0.35, color=C["blue"]).encode(
            x=alt.X("start:T", title="Departure time (local), bar length = flight duration", axis=alt.Axis(format="%H:%M")),
            x2="end:T", y=y, tooltip=tips)
        moved = alt.Chart(plane[plane["added"] > 0]).mark_bar(
            cornerRadius=4, color=C["orange"], stroke="white", strokeWidth=2).encode(
            x="new_start:T", x2="new_end:T", y=y, tooltip=tips)
        gantt = (planned + moved).properties(
            height=alt.Step(40),
            title=alt.Title(f"Aircraft {tail} on {rday:%b %d}",
                            subtitle="Faded = schedule, orange = where the flight moves with this delay"))
        st.altair_chart(styled(gantt), width="stretch", theme=None)
        if not hit.empty:
            hit_show = hit.assign(Departs=hit["sched_dep"].dt.strftime("%H:%M"), Flight="DL" + hit["flight_no"].astype(str))
            st.dataframe(hit_show[["Departs", "Flight", "origin", "dest", "slack_minutes", "added_delay"]].rename(
                columns={"origin": "From", "dest": "To", "slack_minutes": "Slack (min)", "added_delay": "Pushed delay (min)"}),
                hide_index=True, width="stretch")

# =====================================================================
# RECOVERY
# =====================================================================
with tab_recover:
    st.subheader("Break something and watch it get re-planned")
    st.caption("The optimizer can swap aircraft between flights, delay flights and, if it has to, cancel them. "
               "It minimizes: delay minutes + 300 per cancellation + 20 per reassigned flight + "
               "120 per plane ending the day in the wrong city.")
    c1, c2, c3 = st.columns(3)
    rdate = c1.date_input("Date", pd.Timestamp("2024-03-15"), min_value=flights["date"].min(),
                          max_value=flights["date"].max(), key="rec_date")
    rchub = c2.selectbox("Hub", ["ATL", "MSP", "DTW", "SLC"], key="rec_hub")
    times = [f"{h:02d}:{m:02d}" for h in range(6, 21) for m in (0, 30)]
    rstart = c3.selectbox("Disruption starts", times, index=times.index("14:00"), key="rec_start")
    c4, c5, c6 = st.columns(3)
    mode = c4.radio("What breaks?", ["aog", "ground"], key="rec_mode",
                    format_func=lambda m: "A plane breaks down" if m == "aog" else "Ground delay: every departure +60 min")
    hours = c5.slider("For how long? (hours)", 1, 6, 5, key="rec_hours")
    pool = c6.slider("Aircraft the optimizer may use", 6, 14, 12, key="rec_pool")

    if st.button("Run recovery", type="primary"):
        st.session_state["rec_args"] = (str(rdate), rchub, rstart, hours, mode, pool, 30.0)

    args = st.session_state.get("rec_args")
    if not args:
        st.info("Pick a scenario and press **Run recovery**. The solver takes up to a minute while it proves its plan is the best possible.")
    else:
        try:
            with st.spinner("Solving: searching for the best plan and proving it..."):
                plans, info, ready = run_recovery(*args)
        except ValueError as e:
            st.error(str(e))
            plans = None
        if plans:
            what = (f"aircraft **{info['aog_tail']}** stuck at {info['hub']} for {info['aog_hours']:g}h"
                    if info["aog_tail"] else f"every {info['hub']} departure delayed {info['ground_delay']} min")
            st.write(f"{info['date']} from {info['start']}: {what}. "
                     f"Pool: {info['aircraft']} aircraft, {info['flights']} flights.")

            labels = {"do_nothing": "Do nothing (wait)", "delay_or_cancel_only": "Delay / cancel, no swaps",
                      "skyrecover": "SkyRecover (swaps allowed)"}
            rows = []
            for k, p in plans.items():
                if p["summary"] is None:
                    continue
                s = p["summary"]
                rows.append({"Strategy": labels[k], "Delay (min)": s["delay_minutes"], "Cancelled": s["cancelled"],
                             "Reassigned flights": s["flights_reassigned"], "Score": s["score"],
                             "Gap to proven best": p["gap"]})
            res = pd.DataFrame(rows)
            base = plans["do_nothing"]["summary"]["score"]
            best = plans["skyrecover"]["summary"]["score"] if plans["skyrecover"]["summary"] else base
            k1, k2, k3 = st.columns(3)
            k1.metric("Do nothing", f"{base:,}")
            k2.metric("SkyRecover", f"{best:,}", f"{-(base - best) / base:.0%}" if base else None, delta_color="inverse")
            sk = plans["skyrecover"]["summary"]
            k3.metric("Cancellations avoided", plans["delay_or_cancel_only"]["summary"]["cancelled"] - sk["cancelled"])
            st.dataframe(res, hide_index=True, width="stretch",
                         column_config={"Gap to proven best": st.column_config.NumberColumn(
                             format="percent", help="0% = mathematically proven best plan")})

            def gantt_data(table, plan_name):
                t = table[~table["cancelled"]].copy()
                t["plane"] = t["new_tail"]
                t["start"] = pd.Timestamp(info["date"]) + pd.to_timedelta(t["new_dep"], unit="m")
                t["change"] = "On time"
                t.loc[t["delay"] >= 15, "change"] = "Delayed 15+ min"
                t.loc[(t["new_tail"] != t["tail"]) & (t["delay"] < 15), "change"] = "Moved to another plane"
                t["plan"] = plan_name
                return t

            sched_lookup = flights.set_index("flight_id")
            g = pd.concat([gantt_data(plans["do_nothing"]["table"], "Do nothing"),
                           gantt_data(plans["skyrecover"]["table"], "SkyRecover")])
            block = pd.to_timedelta(sched_lookup.loc[g["flight_id"], "block_minutes"].to_numpy(), unit="m")
            g["end"] = g["start"] + block   # real duration, not local landing time (time zones!)
            g["label"] = "DL" + g["flight_no"].astype(str) + " " + g["origin"] + "->" + g["dest"]
            if info["aog_tail"]:
                g.loc[g["plane"] == info["aog_tail"], "plane"] = f"{info['aog_tail']} (broken)"
            kinds = ["On time", "Moved to another plane", "Delayed 15+ min"]
            chart = alt.Chart(g).mark_bar(cornerRadius=4, height=16, stroke="white", strokeWidth=1).encode(
                x=alt.X("start:T", title="Departure time (local)", axis=alt.Axis(format="%H:%M")),
                x2="end:T",
                y=alt.Y("plane:N", title=None),
                color=alt.Color("change:N", title=None, sort=kinds,
                                scale=alt.Scale(domain=kinds, range=[C["blue"], C["aqua"], C["orange"]]),
                                legend=alt.Legend(orient="top")),
                tooltip=[alt.Tooltip("label:N", title="Flight"), alt.Tooltip("plane:N", title="Flown by"),
                         alt.Tooltip("tail:N", title="Originally"), alt.Tooltip("start:T", title="Departs", format="%H:%M"),
                         alt.Tooltip("delay:Q", title="Delay (min)"), alt.Tooltip("change:N", title="")],
            ).properties(height=34 * info["aircraft"], width=420).facet(
                column=alt.Column("plan:N", title=None, sort=["Do nothing", "SkyRecover"]))
            st.altair_chart(styled(chart), theme=None)

            cancelled_dc = plans["delay_or_cancel_only"]["table"]
            if cancelled_dc is not None and cancelled_dc["cancelled"].any():
                cx = cancelled_dc[cancelled_dc["cancelled"]]
                st.caption("Without swaps, the best plan cancels: " + ", ".join(
                    f"DL{r.flight_no} {r.origin}->{r.dest} ({hhmm(r.sched_dep)})" for r in cx.itertuples()))

            sky = plans["skyrecover"]["table"]
            changed = sky[(sky["delay"] > 0) | sky["cancelled"] | (sky["new_tail"] != sky["tail"])].copy()
            changed["Departs"] = changed["sched_dep"].map(hhmm)
            changed["New time"] = changed["new_dep"].map(lambda m: "cancelled" if pd.isna(m) else hhmm(m))
            with st.expander(f"Every change in the SkyRecover plan ({len(changed)} flights)"):
                st.dataframe(changed[["flight_no", "origin", "dest", "Departs", "New time", "delay", "tail", "new_tail"]].rename(
                    columns={"flight_no": "Flight", "origin": "From", "dest": "To", "delay": "Delay (min)",
                             "tail": "Planned aircraft", "new_tail": "New aircraft"}), hide_index=True, width="stretch")

# =====================================================================
# HOW WELL IT PREDICTS
# =====================================================================
with tab_model:
    st.subheader("Tested on a month it never saw")
    d = metrics["data"]
    st.write(f"Trained on {d['train_flights']:,} flights ({d['train_range']}), tested on "
             f"**{d['test_flights']:,} flights** ({d['test_range']}). Every input is something an airline "
             "would know 60 minutes before departure, so the model can't peek at the future.")
    names = {"always_on_time": "Always guess on time", "airport_hour_average": "Typical delay for that airport and hour",
             "ripple_formula": "Slack math alone, no machine learning", "planning": "SkyRecover, schedule only",
             "day_of": "SkyRecover, schedule plus live signals"}
    tbl = pd.DataFrame([{"Approach": names[k], "Avg error (min)": m["mae_minutes"], "AUC": m["auc"],
                         "Right when it warns": m["precision"], "Late flights caught": m["recall"]}
                        for k, m in metrics["models"].items()])
    st.dataframe(tbl, hide_index=True, width="stretch",
                 column_config={"Right when it warns": st.column_config.NumberColumn(format="percent"),
                                "Late flights caught": st.column_config.NumberColumn(format="percent"),
                                "AUC": st.column_config.NumberColumn(format="%.2f", help="0.5 is a coin flip, 1.0 is perfect")})
    st.markdown(
        "**How the pieces fit.** The ripple engine follows each aircraft by tail number: "
        "`slack = ground time - 35 min`, and a late arrival pushes `max(0, delay - slack)` onto the next flight. "
        "The forecaster (LightGBM) combines that with airport conditions and schedule patterns. "
        "The recovery optimizer (Google OR-Tools) re-assigns aircraft and re-times flights to minimize total disruption.")

st.html("""<div class="sr-foot">Flight data: US Bureau of Transportation Statistics, Reporting Carrier On-Time Performance,
Delta Air Lines, January to March 2024. Times are local to each airport.
Built by Daksh Kumar &middot; <a href="https://github.com/DaKSh32-blip/SkyRecover" target="_blank">source on GitHub</a></div>""")
