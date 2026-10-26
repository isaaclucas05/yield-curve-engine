# Portfolio immunization and hedge sizing optimizer.
# Solves for DV01 and joint DV01/Convexity allocations via linear systems.

import numpy as np

try:
    from src.bond_pricing import FixedRateBond, BondRiskMetrics
except ModuleNotFoundError:
    from bond_pricing import FixedRateBond, BondRiskMetrics


# Represents a holding in a specific bond
class PortfolioPosition:
    def __init__(self, bond, notional, dirty_price, accrued_interest=0.0):
        self.bond = bond
        self.notional = float(notional)
        self.dirty_price = float(dirty_price)
        self.accrued_interest = float(accrued_interest)


# Contains hedge notionals and pre/post risk exposures
class ImmunizationResult:
    def __init__(
        self,
        hedge_notionals,
        pre_hedge_dv01,
        post_hedge_dv01,
        pre_hedge_convexity_dollar,
        post_hedge_convexity_dollar,
        net_krd,
    ):
        self.hedge_notionals = hedge_notionals
        self.pre_hedge_dv01 = pre_hedge_dv01
        self.post_hedge_dv01 = post_hedge_dv01
        self.pre_hedge_convexity_dollar = pre_hedge_convexity_dollar
        self.post_hedge_convexity_dollar = post_hedge_convexity_dollar
        self.net_krd = net_krd


# Solves for rate immunization allocations against DV01 and convexity risk
class PortfolioHedgeOptimizer:
    @staticmethod
    def aggregate_portfolio_risk(positions):
        total_value = 0.0
        total_dv01 = 0.0
        total_conv_dollar = 0.0
        total_krds = {}

        for pos in positions:
            metrics = pos.bond.compute_risk_metrics(
                dirty_price=pos.dirty_price,
                accrued_interest=pos.accrued_interest,
            )

            quantity = pos.notional / pos.bond.face_value
            pos_mv = quantity * pos.dirty_price
            total_value += pos_mv

            pos_dv01 = quantity * metrics.dv01
            total_dv01 += pos_dv01

            # Dollar Convexity: 0.5 * Market Value * Convexity * (0.0001)^2
            pos_conv_dollar = 0.5 * pos_mv * metrics.convexity * (1e-4 ** 2)
            total_conv_dollar += pos_conv_dollar

            # Aggregate Key Rate Durations weighted by dollar DV01 contribution
            for tenor, krd in metrics.key_rate_durations.items():
                total_krds[tenor] = total_krds.get(tenor, 0.0) + (pos_dv01 * krd)

        return total_value, total_dv01, total_conv_dollar, total_krds

    # Solves for single-instrument DV01 neutralization: h1 = - DV01_target / dv01_per_dollar_hedge
    def solve_dv01_hedge(self, target_positions, hedge_instrument, hedge_label="Hedge_10Y"):
        _, target_dv01, target_conv_dlr, target_krds = self.aggregate_portfolio_risk(target_positions)

        # Hedge instrument sensitivity per $1 notional
        hedge_metrics = hedge_instrument.bond.compute_risk_metrics(
            dirty_price=hedge_instrument.dirty_price,
            accrued_interest=hedge_instrument.accrued_interest,
        )
        dv01_per_dollar = hedge_metrics.dv01 / hedge_instrument.bond.face_value

        if abs(dv01_per_dollar) < 1e-10:
            raise ValueError("Hedge instrument has zero DV01 sensitivity.")

        hedge_notional = -target_dv01 / dv01_per_dollar

        hedge_position = PortfolioPosition(
            bond=hedge_instrument.bond,
            notional=hedge_notional,
            dirty_price=hedge_instrument.dirty_price,
            accrued_interest=hedge_instrument.accrued_interest,
        )

        _, post_dv01, post_conv_dlr, post_krds = self.aggregate_portfolio_risk(
            target_positions + [hedge_position]
        )

        return ImmunizationResult(
            hedge_notionals={hedge_label: float(hedge_notional)},
            pre_hedge_dv01=float(target_dv01),
            post_hedge_dv01=float(post_dv01),
            pre_hedge_convexity_dollar=float(target_conv_dlr),
            post_hedge_convexity_dollar=float(post_conv_dlr),
            net_krd=post_krds,
        )

    # Solves 2x2 linear system for simultaneous DV01 and Convexity neutralization:
    # [d1  d2] [h1] = [-DV01_target]
    # [c1  c2] [h2]   [-Conv$_target]
    def solve_dv01_convexity_hedge(
        self,
        target_positions,
        hedge_1,
        hedge_2,
        label_1="Hedge_2Y",
        label_2="Hedge_10Y",
    ):
        _, target_dv01, target_conv_dlr, _ = self.aggregate_portfolio_risk(target_positions)

        m1 = hedge_1.bond.compute_risk_metrics(hedge_1.dirty_price, hedge_1.accrued_interest)
        d1 = m1.dv01 / hedge_1.bond.face_value
        c1 = 0.5 * (hedge_1.dirty_price / hedge_1.bond.face_value) * m1.convexity * (1e-4 ** 2)

        m2 = hedge_2.bond.compute_risk_metrics(hedge_2.dirty_price, hedge_2.accrued_interest)
        d2 = m2.dv01 / hedge_2.bond.face_value
        c2 = 0.5 * (hedge_2.dirty_price / hedge_2.bond.face_value) * m2.convexity * (1e-4 ** 2)

        A = np.array([[d1, d2], [c1, c2]], dtype=float)
        b = np.array([-target_dv01, -target_conv_dlr], dtype=float)

        det = np.linalg.det(A)
        if abs(det) < 1e-12:
            raise ValueError(
                f"Singular sensitivity matrix (det={det:.2e}). "
                "The two hedge instruments do not have sufficient convexity separation."
            )

        h1, h2 = np.linalg.solve(A, b)

        pos_h1 = PortfolioPosition(hedge_1.bond, float(h1), hedge_1.dirty_price, hedge_1.accrued_interest)
        pos_h2 = PortfolioPosition(hedge_2.bond, float(h2), hedge_2.dirty_price, hedge_2.accrued_interest)

        _, post_dv01, post_conv_dlr, post_krds = self.aggregate_portfolio_risk(
            target_positions + [pos_h1, pos_h2]
        )

        return ImmunizationResult(
            hedge_notionals={label_1: float(h1), label_2: float(h2)},
            pre_hedge_dv01=float(target_dv01),
            post_hedge_dv01=float(post_dv01),
            pre_hedge_convexity_dollar=float(target_conv_dlr),
            post_hedge_convexity_dollar=float(post_conv_dlr),
            net_krd=post_krds,
        )


if __name__ == "__main__":
    # Test Scenario: $10,000,000 Portfolio of 7-Year Treasuries
    target_bond = FixedRateBond(maturity=7.0, coupon_rate=0.0375, face_value=100.0)
    target_pos = PortfolioPosition(bond=target_bond, notional=10_000_000.0, dirty_price=97.50)

    # Available Liquid Benchmark Hedging Instruments
    bond_2y = FixedRateBond(maturity=2.0, coupon_rate=0.0425, face_value=100.0)
    bond_10y = FixedRateBond(maturity=10.0, coupon_rate=0.0400, face_value=100.0)

    h_2y = PortfolioPosition(bond=bond_2y, notional=100.0, dirty_price=99.20)
    h_10y = PortfolioPosition(bond=bond_10y, notional=100.0, dirty_price=98.50)

    optimizer = PortfolioHedgeOptimizer()

    print("=== CASE 1: PURE DV01 NEUTRALIZATION (USING 10Y NOTE) ===")
    res_dv01 = optimizer.solve_dv01_hedge(
        target_positions=[target_pos],
        hedge_instrument=h_10y,
        hedge_label="US_10Y_BENCHMARK",
    )
    for k, v in res_dv01.hedge_notionals.items():
        print(f"  Target Hedge Notional ({k}): ${v:,.2f}")
    print(f"  Initial Portfolio DV01:          ${res_dv01.pre_hedge_dv01:,.2f}")
    print(f"  Post-Hedge Net DV01:             ${res_dv01.post_hedge_dv01:,.4f}")
    print(f"  Post-Hedge Dollar Convexity:     ${res_dv01.post_hedge_convexity_dollar:,.4f}")

    print("\n=== CASE 2: DUAL DV01 & CONVEXITY NEUTRALIZATION (2Y + 10Y PAIR) ===")
    res_dual = optimizer.solve_dv01_convexity_hedge(
        target_positions=[target_pos],
        hedge_1=h_2y,
        hedge_2=h_10y,
        label_1="US_2Y_NOTE",
        label_2="US_10Y_NOTE",
    )
    for k, v in res_dual.hedge_notionals.items():
        print(f"  Target Hedge Notional ({k}): ${v:,.2f}")
    print(f"  Initial Portfolio DV01:          ${res_dual.pre_hedge_dv01:,.2f}")
    print(f"  Post-Hedge Net DV01:             ${res_dual.post_hedge_dv01:,.4f}")
    print(f"  Initial Dollar Convexity:        ${res_dual.pre_hedge_convexity_dollar:,.4f}")
    print(f"  Post-Hedge Dollar Convexity:     ${res_dual.post_hedge_convexity_dollar:,.6f}")