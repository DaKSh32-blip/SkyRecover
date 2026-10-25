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

st.title("✈️ SkyRecover")
st.caption("Predicts airline delays, shows how one late plane ripples through the network, "
           "and re-plans the day when something breaks. Built on 227,855 real Delta flights "
           "(US DOT data, Jan to Mar 2024).")

tab_over, tab_day, tab_ripple, tab_recover = st.tabs(
    ["Overview", "Day explorer", "Ripple", "Recovery"])

# =====================================================================
# OVERVIEW
# =====================================================================
with tab_over:
    rv = metrics["ripple_validation"]
    day_of = metrics["models"]["day_of"]
    ripple_only = metrics["models"]["ripple_formula"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Ripple model vs. Delta's own reports", f"{rv['correlation']:.2f}",
              help="Correlation with the 'Late Aircraft' delay Delta reports to the government")
    c2.metric("Forecaster AUC on unseen March flights", f"{day_of['auc']:.2f}",
              help="0.5 = coin flip, 1.0 = perfect")
    c3.metric("Late flights caught", f"{day_of['recall']:.0%}",
              f"{day_of['recall'] / ripple_only['recall']:.1f}x the ripple formula alone")

    st.subheader("How good is the forecaster?")
    d = metrics["data"]
    st.write(f"Trained on {d['train_flights']:,} flights ({d['train_range']}), tested on "
             f"**{d['test_flights']:,} flights it never saw** ({d['test_range']}). "
             "Every input is something an airline would know 60 minutes before departure.")
    names = {"always_on_time": "Always guess on time", "airport_hour_average": "Typical delay for airport + hour",
             "ripple_formula": "Ripple formula alone (no ML)", "planning": "SkyRecover, schedule only",
             "day_of": "SkyRecover, schedule + live"}
    tbl = pd.DataFrame([{"Model": names[k], "Avg error (min)": m["mae_minutes"], "AUC": m["auc"],
                         "Precision": m["precision"], "Recall": m["recall"]}
                        for k, m in metrics["models"].items()])
    st.dataframe(tbl, hide_index=True, width="stretch",
                 column_config={"Precision": st.column_config.NumberColumn(format="percent"),
                                "Recall": st.column_config.NumberColumn(format="percent")})

    st.subheader("How it works")
    st.markdown(
        "1. **Ripple engine** follows each aircraft by tail number. "
        "`slack = ground time - 35 min turn`, and a late arrival pushes `max(0, delay - slack)` onto the next flight.\n"
        "2. **Forecaster** (LightGBM) predicts late departures an hour ahead from schedule and live signals.\n"
        "3. **Recovery optimizer** (Google OR-Tools) re-assigns aircraft, re-times and, if needed, "
        "cancels flights to minimize total disruption.\n\n"
        "Code, tests and write-up: [github.com/DaKSh32-blip/SkyRecover](https://github.com/DaKSh32-blip/SkyRecover)")

# =====================================================================
# DAY EXPLORER
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
        st.altair_chart(chart, width="stretch")

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
        st.altair_chart(gantt, width="stretch")
        if not hit.empty:
            hit_show = hit.assign(Departs=hit["sched_dep"].dt.strftime("%H:%M"), Flight="DL" + hit["flight_no"].astype(str))
            st.dataframe(hit_show[["Departs", "Flight", "origin", "dest", "slack_minutes", "added_delay"]].rename(
                columns={"origin": "From", "dest": "To", "slack_minutes": "Slack (min)", "added_delay": "Pushed delay (min)"}),
                hide_index=True, width="stretch")

# =====================================================================
# RECOVERY
# =====================================================================
with tab_recover:
    st.subheader("Break something, and watch it get fixed")
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
            st.altair_chart(chart)

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
