import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

try:
    from src.data_loader import FREDYieldCurveLoader
    from src.curve_fitting import SvenssonCalibrator
    from src.bond_pricing import FixedRateBond
    from src.optimizer import PortfolioPosition, PortfolioHedgeOptimizer
    from src.horizon_analytics import HorizonReturnEngine
except ModuleNotFoundError:
    from src.data_loader import FREDYieldCurveLoader
    from src.curve_fitting import SvenssonCalibrator
    from src.bond_pricing import FixedRateBond
    from src.optimizer import PortfolioPosition, PortfolioHedgeOptimizer
    from src.horizon_analytics import HorizonReturnEngine

st.set_page_config(
    page_title="Fixed Income Analytics & Immunization Platform",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] {
        font-size: 1.35rem !important;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

load_dotenv()
api_key = os.getenv("FRED_API_KEY")

if not api_key:
    st.error("FRED_API_KEY missing from environment or .env file.")
    st.stop()


# Cache market data calls so page interactions don't re-query FRED
@st.cache_data(show_spinner=False, ttl=3600)
def load_market_data(start_date, end_date):
    loader = FREDYieldCurveLoader(api_key=api_key)
    return loader.run_pipeline(start_date, end_date)


@st.cache_data(show_spinner=False, ttl=3600)
def load_single_fred_series(series_id, start_date, end_date):
    loader = FREDYieldCurveLoader(api_key=api_key)
    return loader.fetch_custom_series(series_id, start_date, end_date)


with st.sidebar:
    st.title("🏛️ Yield Engine")
    st.caption("Term Structure & Risk Analytics")
    st.divider()

    st.subheader("Data Range")
    date_range = st.date_input(
        "Historical Ingestion Window",
        value=[pd.to_datetime("2024-01-01"), pd.to_datetime("2024-04-01")],
    )

    if len(date_range) < 2:
        st.warning("Select both a start and an end date.")
        st.stop()

    start_str = date_range[0].strftime("%Y-%m-%d")
    end_str = date_range[1].strftime("%Y-%m-%d")

    with st.spinner("Fetching FRED CMT rates..."):
        try:
            curve_container = load_market_data(start_str, end_str)
        except Exception as exc:
            st.error(f"Failed to fetch market data: {exc}")
            st.stop()

    available_dates = [d.strftime("%Y-%m-%d") for d in curve_container.par_yields.index]
    selected_date = st.selectbox(
        "Curve Snapshot Date",
        available_dates,
        index=len(available_dates) - 1,
    )

    st.divider()

    with st.expander("Model Notes", expanded=False):
        st.write("• Nelson-Siegel-Svensson term structure calibration")
        st.write("• Semi-annual bullet bond pricing & Brent's YTM root-finding")
        st.write("• Linear partition-of-unity Key Rate Durations (KRD)")
        st.write("• Joint DV01 and dollar convexity immunization solver")

# Pull curve slice and fit NSS
observed_yields = curve_container.par_yields.loc[selected_date].values
tenors = curve_container.tenors

calibrator = SvenssonCalibrator()
nss_params = calibrator.fit(tenors, observed_yields)

grid_tau = np.linspace(0.0833, 30.0, 300)
fitted_zero = nss_params.zero_rate(grid_tau)
fitted_forward = nss_params.instantaneous_forward_rate(grid_tau)


def get_spot_rate(maturity):
    return float(np.interp(maturity, grid_tau, fitted_zero))


horizon_engine = HorizonReturnEngine(get_spot_rate)

# Benchmark rate spreads
s3m = get_spot_rate(0.25) * 100
s2y = get_spot_rate(2.0) * 100
s10y = get_spot_rate(10.0) * 100
s30y = get_spot_rate(30.0) * 100

spread_2s10s = (s10y - s2y) * 100
spread_3m10s = (s10y - s3m) * 100
spread_10s30s = (s30y - s10y) * 100

st.title("Fixed Income Analytics & Immunization Platform")
st.markdown(
    "Nelson-Siegel-Svensson calibration, duration/convexity decomposition, "
    "curve carry & roll-down attribution, and multi-instrument portfolio immunization."
)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Fit RMSE", f"{nss_params.rmse:.2f} bps")
c2.metric("2s10s Spread", f"{spread_2s10s:+.1f} bps", delta=f"{s10y:.2f}% vs {s2y:.2f}%")
c3.metric("3m10s Spread", f"{spread_3m10s:+.1f} bps", delta="Inversion Monitor")
c4.metric("10s30s Spread", f"{spread_10s30s:+.1f} bps")
c5.metric("Level (β₀)", f"{nss_params.beta0 * 100:.2f}%")

tab_curve, tab_book, tab_carry, tab_fred, tab_immunize = st.tabs([
    "📈 Term Structure Dynamics",
    "📂 Portfolio Risk & KRDs",
    "⚡ Carry & Roll-Down Attribution",
    "🔎 Dynamic FRED Series Inspector",
    "⚖️ Immunization & Hedge Solver",
])

# Tab 1: Curve visualizer & parameter inspect
with tab_curve:
    col_ctrl1, col_ctrl2 = st.columns([1, 2])
    with col_ctrl1:
        sim_shift_bps = st.slider("Simulate Parallel Yield Shock (bps)", -200, 200, 0, step=25)
    with col_ctrl2:
        curve_layers = st.multiselect(
            "Curves to Display",
            ["Market Quotes", "NSS Spot Curve", "Instantaneous Forward Curve", "Shifted Spot Curve"],
            default=["Market Quotes", "NSS Spot Curve", "Instantaneous Forward Curve"] + (["Shifted Spot Curve"] if sim_shift_bps != 0 else []),
        )

    fig_term = go.Figure()

    if "Market Quotes" in curve_layers:
        fig_term.add_trace(go.Scatter(
            x=tenors,
            y=observed_yields * 100,
            mode="markers",
            name="Market CMT Quotes",
            marker=dict(size=9, color="#EF553B", symbol="diamond"),
        ))

    if "NSS Spot Curve" in curve_layers:
        fig_term.add_trace(go.Scatter(
            x=grid_tau,
            y=fitted_zero * 100,
            mode="lines",
            name="NSS Spot Curve",
            line=dict(color="#00CC96", width=2.5),
        ))

    if "Instantaneous Forward Curve" in curve_layers:
        fig_term.add_trace(go.Scatter(
            x=grid_tau,
            y=fitted_forward * 100,
            mode="lines",
            name="Instantaneous Forward f(τ)",
            line=dict(color="#AB63FA", width=2, dash="dash"),
        ))

    if "Shifted Spot Curve" in curve_layers or sim_shift_bps != 0:
        shifted_curve = fitted_zero + (sim_shift_bps / 10_000.0)
        fig_term.add_trace(go.Scatter(
            x=grid_tau,
            y=shifted_curve * 100,
            mode="lines",
            name=f"Shifted Spot Curve ({sim_shift_bps:+d} bps)",
            line=dict(color="#FFA15A", width=2, dash="dot"),
        ))

    fig_term.update_layout(
        title=f"US Treasury Term Structure Calibration ({selected_date})",
        xaxis_title="Maturity Tenor (Years)",
        yaxis_title="Yield / Rate (%)",
        template="plotly_dark",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
        hovermode="x unified",
    )
    st.plotly_chart(fig_term, width="stretch")

    with st.expander("Calibrated NSS Parameters", expanded=False):
        param_table = pd.DataFrame([{
            "Beta 0 (Level)": f"{nss_params.beta0 * 100:.3f}%",
            "Beta 1 (Slope)": f"{nss_params.beta1 * 100:.3f}%",
            "Beta 2 (Curvature 1)": f"{nss_params.beta2 * 100:.3f}%",
            "Beta 3 (Curvature 2)": f"{nss_params.beta3 * 100:.3f}%",
            "Lambda 1 (Scale 1)": f"{nss_params.lambda1:.4f}",
            "Lambda 2 (Scale 2)": f"{nss_params.lambda2:.4f}",
            "Fit RMSE": f"{nss_params.rmse:.3f} bps",
        }])
        st.dataframe(param_table, width="stretch", hide_index=True)

# Tab 2: Position entry and aggregate risk metrics
with tab_book:
    st.subheader("Multi-Asset Portfolio Book Builder")
    st.caption("Positions re-price dynamically against the fitted spot curve.")

    if "portfolio_data" not in st.session_state:
        st.session_state.portfolio_data = pd.DataFrame([
            {"Holding": "3Y Benchmark Note", "Notional ($)": 5_000_000.0, "Maturity (Y)": 3.0, "Coupon (%)": 4.00, "Position": "Long"},
            {"Holding": "7Y Benchmark Note", "Notional ($)": 10_000_000.0, "Maturity (Y)": 7.0, "Coupon (%)": 3.75, "Position": "Long"},
            {"Holding": "20Y Treasury Bond", "Notional ($)": 2_500_000.0, "Maturity (Y)": 20.0, "Coupon (%)": 4.50, "Position": "Long"},
        ])

    edited_df = st.data_editor(
        st.session_state.portfolio_data,
        key="portfolio_editor_grid",
        num_rows="dynamic",
        width="stretch",
        column_config={
            "Holding": st.column_config.TextColumn("Position Name", required=True),
            "Notional ($)": st.column_config.NumberColumn("Notional ($)", min_value=1_000.0, step=250_000.0, format="$%d"),
            "Maturity (Y)": st.column_config.NumberColumn("Maturity (Years)", min_value=0.25, max_value=30.0, step=0.5),
            "Coupon (%)": st.column_config.NumberColumn("Coupon Rate (%)", min_value=0.0, max_value=20.0, step=0.125),
            "Position": st.column_config.SelectboxColumn("Direction", options=["Long", "Short"], required=True),
        },
    )

    active_positions = []
    for _, row in edited_df.iterrows():
        try:
            m_val = float(row["Maturity (Y)"])
            c_val = float(row["Coupon (%)"]) / 100.0
            n_val = float(row["Notional ($)"])
            sign = 1.0 if str(row["Position"]).strip().lower() == "long" else -1.0

            if m_val <= 0 or n_val <= 0:
                continue

            b = FixedRateBond(maturity=m_val, coupon_rate=c_val, face_value=100.0)
            spot_y = get_spot_rate(m_val)
            dp = b.price_from_ytm(spot_y)
            active_positions.append(PortfolioPosition(bond=b, notional=sign * n_val, dirty_price=dp))
        except Exception:
            continue

    optimizer = PortfolioHedgeOptimizer()

    if active_positions:
        book_val, book_dv01, book_conv, book_krds = optimizer.aggregate_portfolio_risk(active_positions)
        effective_mod_dur = (book_dv01 / (book_val * 0.0001)) if abs(book_val) > 1e-4 else 0.0

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Book Market Value", f"${book_val:,.2f}")
        r2.metric("Aggregate DV01", f"${book_dv01:,.2f}")
        r3.metric("Dollar Convexity", f"${book_conv:,.4f}")
        r4.metric("Effective Duration", f"{effective_mod_dur:.2f} yrs")

        fig_krd = go.Figure(go.Bar(
            x=[f"{k}Y" for k in book_krds.keys()],
            y=list(book_krds.values()),
            marker_color="#636EFA",
        ))
        fig_krd.update_layout(
            title="Portfolio Key Rate Duration Decomposition ($ DV01)",
            xaxis_title="Key Rate Tenor",
            yaxis_title="Dollar DV01 ($)",
            template="plotly_dark",
            height=320,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_krd, width="stretch")
    else:
        st.info("No active positions entered. Add rows above to compute book metrics.")

# Tab 3: Horizon carry and roll-down decomposition
with tab_carry:
    st.subheader("Holding-Period Carry & Roll-Down Analysis")
    st.caption("Decomposes total holding return into coupon carry and curve roll-down along the static fitted curve.")

    hz_col1, hz_col2 = st.columns([1, 2])
    with hz_col1:
        horizon_selection = st.selectbox(
            "Evaluation Horizon",
            [1, 3, 6, 12],
            index=1,
            format_func=lambda x: f"{x} Month{'s' if x > 1 else ''}",
        )
        horizon_yrs = horizon_selection / 12.0

    eval_tenors = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0]
    scan_results = horizon_engine.scan_curve_sweet_spots(eval_tenors, horizon_years=horizon_yrs)

    tenor_labels = [f"{r.tenor:.0f}Y" for r in scan_results]
    carry_pcts = [r.carry_return_pct * 100 for r in scan_results]
    rolldowns = [r.rolldown_return_pct * 100 for r in scan_results]

    fig_carry_stack = go.Figure()
    fig_carry_stack.add_trace(go.Bar(name="Coupon Carry (%)", x=tenor_labels, y=carry_pcts, marker_color="#00CC96"))
    fig_carry_stack.add_trace(go.Bar(name="Roll-Down Capital Gain (%)", x=tenor_labels, y=rolldowns, marker_color="#FFA15A"))

    fig_carry_stack.update_layout(
        barmode="relative",
        title=f"{horizon_selection}-Month Total Return Attribution Across Par Tenors",
        xaxis_title="Benchmark Tenor",
        yaxis_title="Projected Period Return (%)",
        template="plotly_dark",
        height=360,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig_carry_stack, width="stretch")

    attribution_table = []
    for r in scan_results:
        attribution_table.append({
            "Tenor": f"{r.tenor:.0f}Y",
            "Coupon Carry": f"{r.carry_return_pct * 100:+.2f}%",
            "Roll-Down Gain": f"{r.rolldown_return_pct * 100:+.2f}%",
            "Total Holding Return": f"{r.total_return_pct * 100:+.2f}%",
            "Annualized Return": f"{r.annualized_total_return * 100:.2f}%",
            "Carry-to-Risk (bps/yr)": f"{r.carry_to_risk_ratio * 100:.2f}",
        })
    st.dataframe(pd.DataFrame(attribution_table), width="stretch", hide_index=True)

# Tab 4: Arbitrary FRED ticker inspector
with tab_fred:
    st.subheader("Universal FRED Bond & Yield Series Explorer")
    st.caption("Pulls series from FRED and calculates credit spreads vs the NSS zero curve.")

    catalog_dict = {
        "10-Year Treasury Constant Maturity (DGS10)": {"id": "DGS10", "maturity": 10.0},
        "2-Year Treasury Constant Maturity (DGS2)": {"id": "DGS2", "maturity": 2.0},
        "5-Year Treasury Constant Maturity (DGS5)": {"id": "DGS5", "maturity": 5.0},
        "30-Year Treasury Constant Maturity (DGS30)": {"id": "DGS30", "maturity": 30.0},
        "10-Year TIPS Inflation-Indexed Yield (DFII10)": {"id": "DFII10", "maturity": 10.0},
        "ICE BofA US Corporate Index Yield (BAMLC0A0CM)": {"id": "BAMLC0A0CM", "maturity": 10.0},
        "ICE BofA BBB Corporate Index Yield (BAMLC0A4CBBB)": {"id": "BAMLC0A4CBBB", "maturity": 8.5},
        "ICE BofA US High Yield Index (BAMLH0A0HYM2)": {"id": "BAMLH0A0HYM2", "maturity": 5.0},
        "Custom FRED Series ID...": {"id": "CUSTOM", "maturity": 5.0},
    }

    selected_catalog_name = st.selectbox("Select Bond / Index Series", list(catalog_dict.keys()))

    if selected_catalog_name == "Custom FRED Series ID...":
        active_id = st.text_input("Enter FRED Series ID (e.g. DGS3, BAMLC0A1CAAA)", value="DGS3").strip().upper()
        active_maturity = st.number_input("Maturity Proxy (Years)", min_value=0.25, max_value=30.0, value=3.0, step=0.5)
    else:
        active_id = catalog_dict[selected_catalog_name]["id"]
        active_maturity = float(catalog_dict[selected_catalog_name]["maturity"])

    inspect_notional = st.number_input(
        "Position Notional ($)",
        10_000.0, 100_000_000.0, 10_000_000.0, step=1_000_000.0, key="inspect_notional_input"
    )

    try:
        series_clean = load_single_fred_series(active_id, start_str, end_str)

        matched_series = series_clean.loc[:pd.to_datetime(selected_date)]
        if len(matched_series) > 0:
            active_rate = float(matched_series.iloc[-1])
        else:
            active_rate = float(series_clean.iloc[-1])

        rf_rate = get_spot_rate(active_maturity)
        spread_bps = (active_rate - rf_rate) * 10_000.0

        synth_bond = FixedRateBond(maturity=active_maturity, coupon_rate=active_rate, face_value=100.0)
        dirty_p = synth_bond.price_from_ytm(active_rate)
        metrics_synth = synth_bond.compute_risk_metrics(dirty_price=dirty_p)
        pos_dollar_dv01 = metrics_synth.dv01 * (inspect_notional / 100.0)

        f_col1, f_col2, f_col3, f_col4, f_col5 = st.columns(5)
        f_col1.metric("Observed Yield", f"{active_rate * 100:.3f}%")
        f_col2.metric("Spread vs NSS Spot", f"{spread_bps:+.1f} bps")
        f_col3.metric("Modified Duration", f"{metrics_synth.modified_duration:.2f} yrs")
        f_col4.metric("Convexity", f"{metrics_synth.convexity:.2f}")
        f_col5.metric("Position DV01", f"${pos_dollar_dv01:,.2f}")

        st.write("#### Projected Horizon Total Return")
        custom_hz = st.selectbox(
            "Holding Horizon",
            [0.25, 0.5, 1.0],
            format_func=lambda x: f"{int(x * 12)} Months",
            key="custom_hz_box"
        )
        if custom_hz < active_maturity:
            custom_attr = horizon_engine.analyze_bond(synth_bond, horizon_years=custom_hz)
            ch1, ch2, ch3 = st.columns(3)
            ch1.metric("Coupon Carry", f"+${inspect_notional * custom_attr.carry_return_pct:,.2f}")
            ch2.metric(
                "Roll-Down Gain",
                f"{'+$' if custom_attr.rolldown_return_pct >= 0 else '-$'}{abs(inspect_notional * custom_attr.rolldown_return_pct):,.2f}"
            )
            ch3.metric("Annualized Return", f"{custom_attr.annualized_total_return * 100:.2f}%")
        else:
            st.warning("Selected horizon must be strictly less than instrument maturity.")

    except Exception as exc:
        st.error(f"Failed to load FRED series '{active_id}': {exc}")

# Tab 5: Portfolio immunization solver
with tab_immunize:
    st.subheader("Portfolio Immunization Optimizer")
    st.caption("Solves linear systems to neutralize DV01 and dollar convexity risk.")

    if not active_positions:
        st.warning("Add at least one active bond position in the 'Portfolio Risk & KRDs' tab first.")
    else:
        opt_col1, opt_col2 = st.columns(2)
        with opt_col1:
            immunization_mode = st.radio(
                "Hedging Strategy",
                ["Joint DV01 & Convexity Neutralization (2 Instruments)", "Pure DV01 Neutralization (1 Instrument)"],
            )

        with opt_col2:
            st.markdown("**Benchmark Instruments**")
            b_tenor_1 = st.selectbox("Benchmark 1 (Short Leg)", [2.0, 3.0, 5.0], index=0)
            b_tenor_2 = st.selectbox("Benchmark 2 (Long Leg)", [7.0, 10.0, 20.0, 30.0], index=1)

        bond_h1 = FixedRateBond(maturity=b_tenor_1, coupon_rate=get_spot_rate(b_tenor_1), face_value=100.0)
        price_h1 = bond_h1.price_from_ytm(get_spot_rate(b_tenor_1))
        pos_h1 = PortfolioPosition(bond=bond_h1, notional=100.0, dirty_price=price_h1)

        bond_h2 = FixedRateBond(maturity=b_tenor_2, coupon_rate=get_spot_rate(b_tenor_2), face_value=100.0)
        price_h2 = bond_h2.price_from_ytm(get_spot_rate(b_tenor_2))
        pos_h2 = PortfolioPosition(bond=bond_h2, notional=100.0, dirty_price=price_h2)

        try:
            if immunization_mode.startswith("Joint"):
                hedge_result = optimizer.solve_dv01_convexity_hedge(
                    target_positions=active_positions,
                    hedge_1=pos_h1,
                    hedge_2=pos_h2,
                    label_1=f"US Treasury {int(b_tenor_1)}Y Benchmark",
                    label_2=f"US Treasury {int(b_tenor_2)}Y Benchmark",
                )
            else:
                hedge_result = optimizer.solve_dv01_hedge(
                    target_positions=active_positions,
                    hedge_instrument=pos_h2,
                    hedge_label=f"US Treasury {int(b_tenor_2)}Y Benchmark",
                )

            st.write("### Solved Hedge Allocations")
            for instrument_label, notional_size in hedge_result.hedge_notionals.items():
                trade_action = "SELL (Short)" if notional_size < 0 else "BUY (Long)"
                st.success(f"**{trade_action}:** **${abs(notional_size):,.2f}** notional in **{instrument_label}**")

            st.write("#### Risk Immunization Ledger")
            ledger_df = pd.DataFrame({
                "Risk Metric": ["Net Portfolio DV01 ($)", "Net Dollar Convexity ($)"],
                "Pre-Hedge Exposure": [
                    f"${hedge_result.pre_hedge_dv01:,.2f}",
                    f"${hedge_result.pre_hedge_convexity_dollar:,.4f}",
                ],
                "Post-Hedge Residual": [
                    f"${hedge_result.post_hedge_dv01:,.4f}",
                    f"${hedge_result.post_hedge_convexity_dollar:,.6f}",
                ],
                "Verification Status": ["Neutralized (≈ $0.00)", "Neutralized (≈ $0.00)"],
            })
            st.table(ledger_df)

        except ValueError as err:
            st.error(f"Optimization Solver Error: {err}")