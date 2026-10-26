# Sovereign Yield Curve Engine & Risk Platform

A fixed-income analytics suite in Python for fitting Treasury yield curves, quantifying portfolio duration/convexity risk, and solving closed-form immunization hedges using live FRED market data.

---

## Quickstart & Setup

### 1. Environment Setup
```bash
git clone https://github.com/isaaclucas05/yield-curve-engine.git
cd yield-curve-engine
python -m venv .venv
```

### 2. Activate Environment
- **Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```
- **macOS / Linux:**
```bash
source .venv/bin/activate
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Key
Create a `.env` file in the root directory
```env
FRED_API_KEY=your_fred_api_key_here
```

### 5. Run the Dashboard
```bash
streamlit run app.py
```
If not done automatically open `http://localhost:8501` in your browser/what streamlit has returned.

---

## Architecture & Modules

### 1. Market Data Pipeline (`src/data_loader.py`)
- Ingests 11 Constant Maturity Treasury (CMT) tenors (1M to 30Y) concurrently using `ThreadPoolExecutor` to eliminate network bottlenecks.
- Normalizes dates against a business-day calendar with a 3-day forward-fill limit (`max_ffill_days=3`) to handle market holidays.
- Enforces complete-slice filtering (`dropna(how='any')`) so calibration models receive complete yield curve data across every date.

### 2. Term Structure Calibration (`src/curve_fitting.py`)
- Fits the 6-parameter Nelson-Siegel-Svensson (NSS) model to convert discrete CMT yields into continuous spot rates $y(\tau)$ and instantaneous forward rates $f(\tau)$.
- Uses SciPy's bounded `L-BFGS-B` optimizer with residual scaling (in basis points) and multi-start retry seeding to avoid local minima.

### 3. Bond Pricing & Risk Engine (`src/bond_pricing.py`)
- Dispatches exact cash-flow schedules generated backward from maturity to avoid calendar drift.
- Solves for YTM via Brent's method (`scipy.optimize.brentq`) for reliable convergence across distressed and negative rate environments.
- Computes analytical Macaulay Duration, Modified Duration, DV01, and discrete convexity.
- Evaluates Key Rate Duration (KRD) using a triangular partition-of-unity kernel such that the sum of all key rate durations equals the bond's total modified duration.

### 4. Horizon Attribution Engine (`src/horizon_analytics.py`)
- Decomposes holding-period returns along static curves into coupon carry and roll-down capital gains across 1M, 3M, 6M, and 12M windows.
- Automates curve scanning across benchmark tenors to isolate sweet spots maximizing the Carry-to-Risk ratio.

### 5. Portfolio Immunization (`src/optimizer.py`)
- Aggregates multi-bond portfolios into total Dollar DV01, Dollar Convexity, and net KRD vectors.
- Supports single-instrument DV01 neutralization.
- Solves a dual-instrument $2 \times 2$ linear system ($Ah = b$) via LU decomposition to simultaneously eliminate both DV01 and dollar convexity, includes matrix determinant checks to prevent singular hedge pairs.

### 6. Interactive Visual Dashboard (`app.py`)
- Built with Streamlit and Plotly to display real-time fitted curves, parallel rate shock sliders, custom FRED spread overlays, and automated trade hedge tickets.

---

## Repository Structure

```text
yield-curve-engine/
├── src/
│   ├── __init__.py             # Package marker
│   ├── data_loader.py          # Concurrent FRED data pipeline
│   ├── curve_fitting.py        # Nelson-Siegel-Svensson calibrator
│   ├── bond_pricing.py         # Pricing, YTM solver, and KRD kernel
│   ├── horizon_analytics.py    # Carry & roll-down attribution
│   └── optimizer.py            # Linear system immunization solver
├── app.py                      # Interactive Streamlit dashboard
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variables template
├── .gitignore                  # Git tracking rules
└── README.md                   # Project documentation
```