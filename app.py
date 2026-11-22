"""AI Road Risk Analyzer - Streamlit dashboard.

Flow: user input -> src.pipeline.analyze() -> (OSRM routes, Open-Meteo weather) ->
Vertex AI endpoint prediction -> comparison -> recommendation -> this UI.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from src.ai_service import get_predictor_from_env
from src.errors import AppError
from src.pipeline import AnalysisResult, analyze
from src.utils import LEVEL_COLORS, LOG, format_distance, format_duration, setup_logging

setup_logging()

st.set_page_config(page_title="AI Road Risk Analyzer", page_icon="🛣️", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem; max-width: 1200px;}
      .hero {padding: 1.2rem 1.4rem; border-radius: 14px;
             background: linear-gradient(120deg, #0f2a43 0%, #1d4e7a 100%); color: #fff; margin-bottom: 1rem;}
      .hero h1 {margin: 0; font-size: 2rem; color: #fff;}
      .hero p {margin: .3rem 0 0 0; opacity: .88;}
      .badge {display:inline-block; padding: 2px 10px; border-radius: 999px; font-weight: 700;
              font-size: .8rem; color: #fff;}
      .route-card {border: 1px solid rgba(128,128,128,.28); border-radius: 12px; padding: .9rem 1rem; height: 100%;}
      .route-card.best {border: 2px solid #2e9e6b;}
      .small {font-size: .85rem; opacity: .8;}
    </style>
    """,
    unsafe_allow_html=True,
)

TONE_ICON = {"good": "🟢", "ok": "🟡", "bad": "🔴"}


def badge(level: str) -> str:
    return f"<span class='badge' style='background:{LEVEL_COLORS[level]}'>{level}</span>"


@st.cache_resource(show_spinner=False)
def load_predictor():
    return get_predictor_from_env()


# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("System status")
    try:
        predictor = load_predictor()
        if predictor.backend_id == "vertex":
            st.success("AI backend: Google Cloud Vertex AI endpoint")
        else:
            st.warning("AI backend: LOCAL development model (not Vertex AI)")
    except AppError as exc:
        predictor = None
        st.error(exc.user_message)
    st.divider()
    st.subheader("About this prototype")
    st.markdown(
        "- The AI model was trained on **synthetic demonstration data**, not real crash records.\n"
        "- **Weather** is live (Open-Meteo). **Traffic** is estimated from time of day. "
        "**Road condition** and **construction** are **simulated** placeholders.\n"
        "- Risk scores are for learning and demonstration only. **Do not use them for real driving decisions.**"
    )

# --------------------------------------------------------------------------- header + form
st.markdown(
    "<div class='hero'><h1>🛣️ AI Road Risk Analyzer</h1>"
    "<p>Compare route alternatives by predicted road risk, not just distance. "
    "Risk scores are predicted by a model hosted on Google Cloud Vertex AI.</p></div>",
    unsafe_allow_html=True,
)

with st.form("route_form"):
    c1, c2 = st.columns(2)
    start_text = c1.text_input("Starting location", value="Bengaluru", placeholder="e.g. Bengaluru")
    end_text = c2.text_input("Destination", value="Mysuru", placeholder="e.g. Mysuru")
    submitted = st.form_submit_button("Analyze Route", type="primary", use_container_width=True)

if submitted:
    st.session_state.pop("result", None)
    st.session_state.pop("error", None)
    if predictor is None:
        st.session_state["error"] = "The AI service is not configured. See the System status panel."
    else:
        try:
            with st.spinner("Finding routes, collecting conditions and asking the AI model..."):
                st.session_state["result"] = analyze(start_text, end_text, predictor)
        except AppError as exc:
            LOG.warning("Analysis failed: %s", exc)
            st.session_state["error"] = exc.user_message
        except Exception:  # never leak internals to the UI
            LOG.exception("Unexpected failure during analysis")
            st.session_state["error"] = "An unexpected error occurred. Please try again."

if "error" in st.session_state:
    st.error(st.session_state["error"])

result: AnalysisResult | None = st.session_state.get("result")


# --------------------------------------------------------------------------- rendering helpers
def render_map(res: AnalysisResult) -> None:
    import folium

    best = res.recommended
    mid = best.coordinates[len(best.coordinates) // 2]
    fmap = folium.Map(location=mid, zoom_start=8, tiles="OpenStreetMap", control_scale=True)
    for r in reversed(res.routes):  # draw best last so it is on top
        label = f"{r.name}: {r.assessment.score:.0f}/100 ({r.assessment.level}), {format_distance(r.distance_km)}"
        folium.PolyLine(
            r.coordinates,
            color=LEVEL_COLORS[r.assessment.level],
            weight=7 if r.is_recommended else 4,
            opacity=0.95 if r.is_recommended else 0.6,
            dash_array=None if r.route_source != "fallback" else "8 8",
            tooltip=label,
        ).add_to(fmap)
    folium.Marker([res.start.lat, res.start.lon], tooltip=f"Start: {res.start.query}",
                  icon=folium.Icon(color="green", icon="play")).add_to(fmap)
    folium.Marker([res.end.lat, res.end.lon], tooltip=f"Destination: {res.end.query}",
                  icon=folium.Icon(color="red", icon="flag")).add_to(fmap)
    lats = [p[0] for r in res.routes for p in r.coordinates]
    lons = [p[1] for r in res.routes for p in r.coordinates]
    fmap.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
    components.html(fmap._repr_html_(), height=480)


def score_chart(res: AnalysisResult) -> go.Figure:
    names = [r.route_id for r in res.routes]
    fig = go.Figure(
        go.Bar(
            x=[f"Route {n}" for n in names],
            y=[r.assessment.score for r in res.routes],
            marker_color=[LEVEL_COLORS[r.assessment.level] for r in res.routes],
            text=[f"{r.assessment.score:.0f}" for r in res.routes],
            textposition="outside",
        )
    )
    fig.add_hline(y=30, line_dash="dot", annotation_text="LOW < 30")
    fig.add_hline(y=55, line_dash="dot", annotation_text="HIGH >= 55")
    fig.update_layout(title="AI risk score by route (lower is safer)", yaxis=dict(range=[0, 105], title="Risk score"),
                      height=360, margin=dict(t=50, b=20, l=10, r=10), showlegend=False)
    return fig


def factor_chart(res: AnalysisResult) -> go.Figure:
    fig = go.Figure()
    for r in res.routes:
        sev = r.features.severity()
        fig.add_trace(go.Bar(name=f"Route {r.route_id}", x=list(sev), y=[v * 100 for v in sev.values()]))
    fig.update_layout(barmode="group", title="Risk factor severity by route (0 = best, 100 = worst)",
                      yaxis=dict(range=[0, 105], title="Severity"), height=360,
                      margin=dict(t=50, b=20, l=10, r=10))
    return fig


def distance_risk_chart(res: AnalysisResult) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=[r.distance_km for r in res.routes], y=[r.assessment.score for r in res.routes],
            mode="markers+text", text=[f"Route {r.route_id}" for r in res.routes], textposition="top center",
            marker=dict(size=18, color=[LEVEL_COLORS[r.assessment.level] for r in res.routes]),
        )
    )
    fig.update_layout(title="Distance vs risk: the shortest route is not always the safest",
                      xaxis_title="Distance (km)", yaxis_title="Risk score", height=360,
                      margin=dict(t=50, b=20, l=10, r=10), showlegend=False)
    return fig


# --------------------------------------------------------------------------- results
if result:
    best = result.recommended

    for w in result.warnings:
        st.warning(w)
    if result.backend_id == "vertex":
        st.success(
            f"Analysis complete. {len(result.routes)} route(s) scored by the Vertex AI endpoint "
            f"in {result.ai_latency_ms:.0f} ms."
        )
    else:
        st.info("Analysis complete using the LOCAL development model (not Vertex AI).")

    st.subheader(f"Recommended: {best.name}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Risk score", f"{best.assessment.score:.0f} / 100")
    m2.markdown(f"**Risk level**<br>{badge(best.assessment.level)}", unsafe_allow_html=True)
    m3.metric("Distance", format_distance(best.distance_km))
    m4.metric("Typical drive time", format_duration(best.duration_min))
    st.info(result.explanation)

    st.subheader("Route comparison")
    cols = st.columns(len(result.routes))
    for col, r in zip(cols, result.routes):
        with col:
            reasons = "".join(f"<li>{TONE_ICON[x.tone]} {x.text}</li>" for x in r.reasons)
            st.markdown(
                f"<div class='route-card {'best' if r.is_recommended else ''}'>"
                f"<b>{r.name}</b><br>{badge(r.assessment.level)} &nbsp; "
                f"<b>{r.assessment.score:.0f}/100</b><br>"
                f"<span class='small'>{format_distance(r.distance_km)} · {format_duration(r.duration_min)}</span><br>"
                f"<b>Recommendation: {r.recommendation}</b>"
                f"<ul style='padding-left:1.1rem;margin:.5rem 0 0 0'>{reasons}</ul></div>",
                unsafe_allow_html=True,
            )

    st.subheader("Map")
    render_map(result)
    if result.used_fallback_routes:
        st.caption("Dashed lines are approximate demo routes, not real roads.")

    st.subheader("Charts")
    ch1, ch2 = st.columns(2)
    ch1.plotly_chart(score_chart(result), use_container_width=True)
    ch2.plotly_chart(factor_chart(result), use_container_width=True)
    st.plotly_chart(distance_risk_chart(result), use_container_width=True)

    with st.expander("Data sources and technical details"):
        rows = []
        for r in result.routes:
            for factor, src in r.data_sources.items():
                key = {"Road condition": "road_condition", "Traffic": "traffic_level", "Weather": "weather_condition",
                       "Construction": "construction", "Road type": "road_type", "Distance": "distance_km"}[factor]
                rows.append({"Route": r.route_id, "Factor": factor, "Value": getattr(r.features, key), "Source": src})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.markdown(
            f"**AI backend:** {result.backend_label}  \n"
            f"**Routes sent in one prediction request:** {len(result.routes)}  \n"
            f"**Prediction latency:** {result.ai_latency_ms:.0f} ms  \n"
            f"**Route source:** {', '.join(sorted({r.route_source for r in result.routes}))}"
        )
else:
    st.caption("Enter two places and click **Analyze Route** to compare routes by predicted risk.")

st.divider()
st.caption(
    "Prototype for learning. Model trained on synthetic data; road condition and construction values are simulated; "
    "traffic is estimated. Not a substitute for real navigation or safety advice."
)
