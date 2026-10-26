# Horizon return, carry, and roll-down analytics engine along static curves.

import numpy as np

try:
    from src.bond_pricing import FixedRateBond
except ModuleNotFoundError:
    from bond_pricing import FixedRateBond


# Holding-period return decomposition over horizon dt
class HorizonReturnAttribution:
    def __init__(
        self,
        tenor,
        horizon_years,
        initial_clean_price,
        initial_dirty_price,
        ending_clean_price,
        ending_dirty_price,
        coupons_collected,
        total_dollar_pnl,
        total_return_pct,
        carry_return_pct,
        rolldown_return_pct,
        annualized_total_return,
        carry_to_risk_ratio,
    ):
        self.tenor = tenor
        self.horizon_years = horizon_years
        self.initial_clean_price = initial_clean_price
        self.initial_dirty_price = initial_dirty_price
        self.ending_clean_price = ending_clean_price
        self.ending_dirty_price = ending_dirty_price
        self.coupons_collected = coupons_collected
        self.total_dollar_pnl = total_dollar_pnl
        self.total_return_pct = total_return_pct
        self.carry_return_pct = carry_return_pct
        self.rolldown_return_pct = rolldown_return_pct
        self.annualized_total_return = annualized_total_return
        self.carry_to_risk_ratio = carry_to_risk_ratio


# Decomposes returns into coupon carry and roll-down capital gains along static curves
class HorizonReturnEngine:
    def __init__(self, spot_curve_fn):
        self.spot_curve_fn = spot_curve_fn

    def analyze_bond(self, bond, horizon_years=0.25):
        if horizon_years >= bond.maturity:
            raise ValueError("Horizon must be strictly less than remaining bond maturity.")

        # Initial bond state at evaluation date
        initial_yield = self.spot_curve_fn(bond.maturity)
        initial_dirty = bond.price_from_ytm(initial_yield)
        initial_clean = initial_dirty
        metrics_t0 = bond.compute_risk_metrics(initial_dirty)

        # Coupon income accrued and paid within holding period window
        coupons_collected = float(
            np.sum(bond.cash_flows[(bond.payment_times > 1e-6) & (bond.payment_times <= horizon_years)])
        )

        # Advance forward along static curve: remaining maturity = T - horizon
        remaining_maturity = bond.maturity - horizon_years
        rolled_bond = FixedRateBond(
            maturity=remaining_maturity,
            coupon_rate=bond.coupon_rate,
            face_value=bond.face_value,
            frequency=bond.frequency,
        )

        # Reprice bond at shifted curve node
        rolled_yield = self.spot_curve_fn(remaining_maturity)
        ending_dirty = rolled_bond.price_from_ytm(rolled_yield)
        ending_clean = ending_dirty

        # Return and PnL decomposition
        total_dollar_pnl = (ending_dirty + coupons_collected) - initial_dirty
        total_return_pct = total_dollar_pnl / initial_dirty
        annualized_total_return = total_return_pct * (1.0 / horizon_years)

        carry_return_pct = coupons_collected / initial_dirty
        rolldown_dollar = ending_clean - initial_clean
        rolldown_return_pct = rolldown_dollar / initial_dirty

        carry_risk = (
            annualized_total_return / metrics_t0.modified_duration
            if metrics_t0.modified_duration > 1e-4
            else 0.0
        )

        return HorizonReturnAttribution(
            tenor=bond.maturity,
            horizon_years=horizon_years,
            initial_clean_price=initial_clean,
            initial_dirty_price=initial_dirty,
            ending_clean_price=ending_clean,
            ending_dirty_price=ending_dirty,
            coupons_collected=coupons_collected,
            total_dollar_pnl=total_dollar_pnl,
            total_return_pct=total_return_pct,
            carry_return_pct=carry_return_pct,
            rolldown_return_pct=rolldown_return_pct,
            annualized_total_return=annualized_total_return,
            carry_to_risk_ratio=carry_risk,
        )

    # Scans par-coupon bonds across the curve to locate maximum roll-down steepness
    def scan_curve_sweet_spots(self, tenor_grid=None, horizon_years=0.25):
        if tenor_grid is None:
            tenor_grid = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0]

        results = []
        for t in tenor_grid:
            if t <= horizon_years:
                continue
            par_rate = self.spot_curve_fn(t)
            bond = FixedRateBond(maturity=t, coupon_rate=par_rate, face_value=100.0)
            res = self.analyze_bond(bond, horizon_years=horizon_years)
            results.append(res)

        return results


if __name__ == "__main__":
    # Test with an upward sloping synthetic curve: y(t) = 0.03 + 0.02 * (1 - exp(-t/3))
    def synthetic_curve(tau: float) -> float:
        return 0.03 + 0.02 * (1.0 - np.exp(-tau / 3.0))

    engine = HorizonReturnEngine(synthetic_curve)
    
    # 5-year par bond held for 6 months (0.5 years)
    test_bond = FixedRateBond(maturity=5.0, coupon_rate=synthetic_curve(5.0), face_value=100.0)
    attr = engine.analyze_bond(test_bond, horizon_years=0.5)

    print("=== 6-MONTH HORIZON RETURN ATTRIBUTION (5Y PAR BOND) ===")
    print(f"Initial Dirty Price:       ${attr.initial_dirty_price:.2f}")
    print(f"Ending Dirty Price:        ${attr.ending_dirty_price:.2f}")
    print(f"Coupons Received:          ${attr.coupons_collected:.2f}")
    print(f"Total Dollar Gain:         ${attr.total_dollar_pnl:.2f}")
    print(f"Carry Return:              +{attr.carry_return_pct * 100:.2f}%")
    print(f"Roll-Down Capital Gain:    +{attr.rolldown_return_pct * 100:.2f}%")
    print(f"Annualized Total Return:   {attr.annualized_total_return * 100:.2f}%")
    print(f"Carry-to-Risk (ModDur):    {attr.carry_to_risk_ratio * 100:.2f} bps/yr")