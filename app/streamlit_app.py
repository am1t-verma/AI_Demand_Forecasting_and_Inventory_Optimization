from __future__ import annotations

import html

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import forecasting as fc


st.set_page_config(
    page_title="Food Demand Forecasting",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    :root {
        --ink: #172033;
        --muted: #64748b;
        --line: #e5eaf2;
        --panel: #ffffff;
        --accent: #2563eb;
        --accent-soft: #eff6ff;
    }
    .stApp {
        background: #fbfcfe;
        color: var(--ink);
    }
    .block-container {
        padding-top: 1.25rem;
        padding-bottom: 2rem;
        max-width: 1220px;
    }
    section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid var(--line);
    }
    .topbar {
        align-items: center;
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        display: flex;
        justify-content: space-between;
        margin-bottom: 1rem;
        padding: 1rem 1.1rem;
    }
    .topbar h1 {
        font-size: 1.75rem;
        line-height: 1.15;
        margin: 0.12rem 0 0;
        letter-spacing: 0;
    }
    .eyebrow {
        color: var(--muted);
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
    }
    .author-pill {
        background: var(--accent-soft);
        border: 1px solid #dbeafe;
        border-radius: 999px;
        color: #1d4ed8;
        font-size: 0.86rem;
        font-weight: 700;
        padding: 0.42rem 0.72rem;
        white-space: nowrap;
    }
    div[data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.75rem 0.9rem;
    }
    div[data-testid="stMetricLabel"] p {
        color: var(--muted);
        font-size: 0.82rem;
    }
    div[data-testid="stMetricValue"] {
        color: var(--ink);
        font-size: 1.55rem;
    }
    .stPlotlyChart {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.35rem;
    }
    .section-title {
        color: var(--ink);
        font-size: 1.05rem;
        font-weight: 800;
        margin: 0.65rem 0 0.45rem;
    }
    .dataframe {
        border-radius: 8px;
    }
    @media (max-width: 760px) {
        .topbar {
            align-items: flex-start;
            flex-direction: column;
            gap: 0.65rem;
        }
        .topbar h1 {
            font-size: 1.45rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_artifacts():
    return fc.load_artifacts()


@st.cache_data(show_spinner=False)
def get_training_data() -> pd.DataFrame:
    return fc.load_training_data()


def float_control(
    label: str,
    min_value: float,
    max_value: float,
    value: float,
    step: float,
    key: str,
) -> float:
    min_value = float(round(min_value, 2))
    max_value = float(round(max_value, 2))
    value = float(round(value, 2))

    if max_value <= min_value:
        return float(st.number_input(label, value=value, step=step, key=key))
    return float(
        st.slider(
            label,
            min_value=min_value,
            max_value=max_value,
            value=min(max(value, min_value), max_value),
            step=step,
            key=key,
        )
    )


def orders_slider(
    label: str,
    value: float,
    max_orders: int,
    key: str,
    disabled: bool,
) -> float:
    return float(
        st.slider(
            label,
            min_value=0,
            max_value=max_orders,
            value=int(min(max(round(value), 0), max_orders)),
            step=1,
            key=key,
            disabled=disabled,
        )
    )


def line_forecast_chart(history: pd.DataFrame, forecast: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history["week"],
            y=history["num_orders"],
            mode="lines+markers",
            name="Historical",
            line=dict(color="#2563eb", width=2),
            marker=dict(size=5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast["week"],
            y=forecast["predicted_orders"],
            mode="lines+markers",
            name="Forecast",
            line=dict(color="#f97316", width=2, dash="dash"),
            marker=dict(size=7),
        )
    )
    fig.add_vline(
        x=int(history["week"].max()),
        line_width=1,
        line_dash="dot",
        line_color="#94a3b8",
    )
    fig.update_layout(
        template="plotly_white",
        height=390,
        margin=dict(l=15, r=15, t=18, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Week",
        yaxis_title="Orders",
    )
    return fig


def forecast_bar_chart(forecast: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        forecast,
        x="week",
        y="predicted_orders",
        text="predicted_orders",
        color="predicted_orders",
        color_continuous_scale=["#dbeafe", "#22c55e"],
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        template="plotly_white",
        height=330,
        margin=dict(l=15, r=15, t=18, b=10),
        coloraxis_showscale=False,
        xaxis_title="Week",
        yaxis_title="Predicted orders",
    )
    return fig


def importance_chart(importance: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        importance,
        x="importance",
        y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale=["#e0f2fe", "#2563eb"],
    )
    fig.update_layout(
        template="plotly_white",
        height=330,
        margin=dict(l=15, r=15, t=18, b=10),
        coloraxis_showscale=False,
        xaxis_title="Importance",
        yaxis_title="",
    )
    return fig


try:
    model, encoder = get_artifacts()
    train_df = get_training_data()
    center_df = fc.center_raw_data(train_df)
except Exception as exc:
    st.error(f"Could not load forecasting pipeline: {exc}")
    st.stop()


summary_df = fc.meal_summary(center_df)
meal_ids = [int(value) for value in summary_df["meal_id"].tolist()]
meal_labels = {
    int(row.meal_id): str(row.label) for row in summary_df.itertuples(index=False)
}
default_meal = int(summary_df.iloc[0]["meal_id"])
encoder_choices = fc.encoder_options(encoder)

with st.sidebar:
    st.markdown("### Pipeline")
    author_name = st.text_input("Author name", value="MrAmi")
    horizon = st.slider("Forecast weeks", min_value=1, max_value=12, value=8, step=1)
    history_window = st.slider("History window", min_value=8, max_value=52, value=26, step=1)

    st.markdown("### Manual Features")
    meal_id = int(
        st.select_slider(
            "Meal id",
            options=meal_ids,
            value=default_meal,
            format_func=lambda value: meal_labels[int(value)],
        )
    )
    profile = fc.latest_profile(center_df, meal_id)

    checkout_price = float_control(
        "Checkout price",
        center_df["checkout_price"].min(),
        center_df["checkout_price"].max(),
        profile["checkout_price"],
        0.5,
        f"checkout_{meal_id}",
    )
    base_price = float_control(
        "Base price",
        center_df["base_price"].min(),
        center_df["base_price"].max(),
        profile["base_price"],
        0.5,
        f"base_{meal_id}",
    )
    op_area = float_control(
        "Operation area",
        center_df["op_area"].min(),
        center_df["op_area"].max(),
        profile["op_area"],
        0.1,
        f"op_area_{meal_id}",
    )

    emailer_for_promotion = int(
        st.toggle(
            "Email promotion",
            value=bool(profile["emailer_for_promotion"]),
            key=f"emailer_{meal_id}",
        )
    )
    homepage_featured = int(
        st.toggle(
            "Homepage featured",
            value=bool(profile["homepage_featured"]),
            key=f"homepage_{meal_id}",
        )
    )

    center_type = st.selectbox(
        "Center type",
        encoder_choices["center_type"],
        index=encoder_choices["center_type"].index(profile["center_type"]),
        key=f"center_type_{meal_id}",
    )
    category = st.selectbox(
        "Category",
        encoder_choices["category"],
        index=encoder_choices["category"].index(profile["category"]),
        key=f"category_{meal_id}",
    )
    cuisine = st.selectbox(
        "Cuisine",
        encoder_choices["cuisine"],
        index=encoder_choices["cuisine"].index(profile["cuisine"]),
        key=f"cuisine_{meal_id}",
    )

    st.markdown("### Demand Memory")
    use_manual_memory = st.toggle("Manual lag controls", value=True, key=f"manual_{meal_id}")
    max_orders = int(max(center_df["num_orders"].max(), profile["historical_max"]) * 1.15)
    lag_1 = orders_slider(
        "Lag 1 orders",
        profile["lag_1"],
        max_orders,
        f"lag1_{meal_id}",
        not use_manual_memory,
    )
    lag_2 = orders_slider(
        "Lag 2 orders",
        profile["lag_2"],
        max_orders,
        f"lag2_{meal_id}",
        not use_manual_memory,
    )
    lag_3 = orders_slider(
        "Lag 3 orders",
        profile["lag_3"],
        max_orders,
        f"lag3_{meal_id}",
        not use_manual_memory,
    )
    lag_7 = orders_slider(
        "Lag 7 orders",
        profile["lag_7"],
        max_orders,
        f"lag7_{meal_id}",
        not use_manual_memory,
    )
    rolling_mean_7 = orders_slider(
        "Rolling mean 7",
        profile["rolling_mean_7"],
        max_orders,
        f"rollmean_{meal_id}",
        not use_manual_memory,
    )
    rolling_std_7 = orders_slider(
        "Rolling std 7",
        profile["rolling_std_7"],
        max_orders,
        f"rollstd_{meal_id}",
        not use_manual_memory,
    )


safe_author = html.escape(author_name.strip() or "MrAmi")
st.markdown(
    f"""
    <div class="topbar">
        <div>
            <div class="eyebrow">Center {fc.CENTER_ID} demand pipeline</div>
            <h1>Food Demand Forecasting</h1>
        </div>
        <div class="author-pill">Author: {safe_author}</div>
    </div>
    """,
    unsafe_allow_html=True,
)


scenario = fc.Scenario(
    meal_id=meal_id,
    checkout_price=checkout_price,
    base_price=base_price,
    emailer_for_promotion=emailer_for_promotion,
    homepage_featured=homepage_featured,
    center_type=center_type,
    category=category,
    cuisine=cuisine,
    op_area=op_area,
    lag_1=lag_1,
    lag_2=lag_2,
    lag_3=lag_3,
    lag_7=lag_7,
    rolling_mean_7=rolling_mean_7,
    rolling_std_7=rolling_std_7,
    use_manual_memory=use_manual_memory,
)

forecast_df, first_features = fc.forecast_orders(
    model=model,
    encoder=encoder,
    center_df=center_df,
    scenario=scenario,
    horizon=horizon,
)

discount_per = fc.discount_percent(base_price, checkout_price)
recent_history = (
    center_df[center_df["meal_id"] == meal_id]
    .sort_values("week")
    .tail(history_window)
)

metric_cols = st.columns(4)
metric_cols[0].metric("Next week", f"{int(forecast_df.iloc[0]['predicted_orders']):,}")
metric_cols[1].metric(
    f"{horizon} week total",
    f"{int(forecast_df['predicted_orders'].sum()):,}",
)
metric_cols[2].metric("Recent avg", f"{profile['recent_avg_4']:.0f}")
metric_cols[3].metric("Discount", f"{discount_per:.1f}%")

st.markdown('<div class="section-title">Forecast Trend</div>', unsafe_allow_html=True)
st.plotly_chart(
    line_forecast_chart(recent_history, forecast_df),
    use_container_width=True,
    config={"displayModeBar": False},
)

left, right = st.columns([1, 1])
with left:
    st.markdown('<div class="section-title">Forecast by Week</div>', unsafe_allow_html=True)
    st.plotly_chart(
        forecast_bar_chart(forecast_df),
        use_container_width=True,
        config={"displayModeBar": False},
    )

with right:
    st.markdown(
        '<div class="section-title">Model Feature Importance</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        importance_chart(fc.feature_importance(model, top_n=14)),
        use_container_width=True,
        config={"displayModeBar": False},
    )

audit_cols = [
    ("Meal id", meal_id),
    ("Week", int(forecast_df.iloc[0]["week"])),
    ("Checkout price", f"{checkout_price:.2f}"),
    ("Base price", f"{base_price:.2f}"),
    ("Discount bin", fc.discount_bin(discount_per)),
    ("Price band", fc.price_band(checkout_price)),
    ("Email promotion", emailer_for_promotion),
    ("Homepage featured", homepage_featured),
    ("Center type", center_type),
    ("Category", category),
    ("Cuisine", cuisine),
    ("Operation area", f"{op_area:.1f}"),
    ("Lag 1", f"{lag_1:.0f}"),
    ("Lag 2", f"{lag_2:.0f}"),
    ("Lag 3", f"{lag_3:.0f}"),
    ("Lag 7", f"{lag_7:.0f}"),
    ("Rolling mean 7", f"{rolling_mean_7:.0f}"),
    ("Rolling std 7", f"{rolling_std_7:.0f}"),
]
audit_df = pd.DataFrame(audit_cols, columns=["Feature", "Value"])

table_left, table_right = st.columns([0.9, 1.1])
with table_left:
    st.markdown('<div class="section-title">Manual Feature Snapshot</div>', unsafe_allow_html=True)
    st.dataframe(audit_df, use_container_width=True, hide_index=True)

with table_right:
    st.markdown('<div class="section-title">Forecast Table</div>', unsafe_allow_html=True)
    st.dataframe(
        forecast_df[
            [
                "week",
                "meal_id",
                "predicted_orders",
                "discount_per",
                "price_band",
                "discount_bin",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "Download forecast CSV",
        data=forecast_df.to_csv(index=False).encode("utf-8"),
        file_name=f"center_{fc.CENTER_ID}_meal_{meal_id}_forecast.csv",
        mime="text/csv",
        use_container_width=True,
    )

top_meals = (
    summary_df.head(10)
    .sort_values("avg_orders")
    .assign(meal_id=lambda data: data["meal_id"].astype(str))
)
fig_top = px.bar(
    top_meals,
    x="avg_orders",
    y="meal_id",
    orientation="h",
    color="cuisine",
    hover_data=["category", "cuisine", "last_orders"],
    color_discrete_sequence=["#2563eb", "#f97316", "#16a34a", "#7c3aed"],
)
fig_top.update_layout(
    template="plotly_white",
    height=330,
    margin=dict(l=15, r=15, t=18, b=10),
    xaxis_title="Average historical orders",
    yaxis_title="Meal id",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.markdown('<div class="section-title">Top Meals at Center 13</div>', unsafe_allow_html=True)
st.plotly_chart(fig_top, use_container_width=True, config={"displayModeBar": False})

with st.expander("Model input row"):
    st.dataframe(first_features, use_container_width=True, hide_index=True)
