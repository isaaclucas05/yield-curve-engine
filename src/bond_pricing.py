# Cash flow schedule, bond pricing, and duration/convexity analytics.

import numpy as np
from scipy.optimize import brentq


# Complete risk profile container
class BondRiskMetrics:
    def __init__(
        self,
        clean_price,
        dirty_price,
        accrued_interest,
        ytm,
        macaulay_duration,
        modified_duration,
        dv01,
        convexity,
        key_rate_durations,
    ):
        self.clean_price = clean_price
        self.dirty_price = dirty_price
        self.accrued_interest = accrued_interest
        self.ytm = ytm
        self.macaulay_duration = macaulay_duration
        self.modified_duration = modified_duration
        self.dv01 = dv01
        self.convexity = convexity
        self.key_rate_durations = key_rate_durations


# Standard fixed coupon bond model
class FixedRateBond:
    def __init__(self, maturity, coupon_rate, face_value=100.0, frequency=2):
        if maturity <= 0:
            raise ValueError("Maturity must be positive.")
        if frequency <= 0:
            raise ValueError("Coupon frequency must be positive.")

        self.maturity = float(maturity)
        self.coupon_rate = float(coupon_rate)
        self.face_value = float(face_value)
        self.frequency = int(frequency)

        self.payment_times, self.cash_flows = self._build_cash_flows()

    def _build_cash_flows(self):
        # Step backward from maturity date to establish periodic coupon schedule
        dt = 1.0 / self.frequency
        times = []
        t = self.maturity
        while t > 1e-6:
            times.append(t)
            t -= dt
        times = np.array(sorted(times), dtype=float)

        # Coupon amount per compounding period + bullet principal repayment
        coupon_payment = (self.coupon_rate * self.face_value) / self.frequency
        cfs = np.full_like(times, coupon_payment)
        cfs[-1] += self.face_value

        return times, cfs

    # Discounts cash flows using periodic compounding: sum(CF_k / (1 + y/m)^(m*t_k))
    def price_from_ytm(self, ytm):
        m = self.frequency
        discount_factors = (1.0 + ytm / m) ** (-m * self.payment_times)
        return float(np.sum(self.cash_flows * discount_factors))

    def price_from_discount_factors(self, discount_factors):
        if len(discount_factors) != len(self.cash_flows):
            raise ValueError("Discount factors length must match cash flows.")
        return float(np.sum(self.cash_flows * discount_factors))

    # Solves for annualized yield-to-maturity via Brent root-finding
    def solve_ytm(self, dirty_price):
        if dirty_price <= 0:
            raise ValueError("Dirty price must be strictly positive.")

        def objective(y):
            return self.price_from_ytm(y) - dirty_price

        try:
            return float(brentq(objective, a=-0.10, b=0.50, xtol=1e-8))
        except ValueError:
            # Fallback range to capture negative yields or distressed scenarios
            return float(brentq(objective, a=-0.50, b=1.50, xtol=1e-8))

    def compute_risk_metrics(self, dirty_price, accrued_interest=0.0, key_tenors=None):
        clean_price = dirty_price - accrued_interest
        ytm = self.solve_ytm(dirty_price)
        m = self.frequency

        # Present value of each cash flow component
        pv_factors = (1.0 + ytm / m) ** (-m * self.payment_times)
        pv_cfs = self.cash_flows * pv_factors

        # Macaulay and Modified Duration
        mac_dur = float(np.sum(self.payment_times * pv_cfs) / dirty_price)
        mod_dur = mac_dur / (1.0 + ytm / m)
        dv01 = dirty_price * mod_dur * 0.0001

        # Analytical convexity under periodic compounding
        conv_weights = self.payment_times * (self.payment_times + 1.0 / m)
        convexity = float(
            np.sum(self.cash_flows * conv_weights * (1.0 + ytm / m) ** (-m * self.payment_times - 2))
            / dirty_price
        )

        # Key Rate Duration profile across specified term structure nodes
        if key_tenors is None:
            key_tenors = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0]

        kts = np.array(sorted(key_tenors), dtype=float)
        krds = {}
        dy = 0.0001  # 1 bp parallel perturbation

        for i, kt in enumerate(kts):
            # Zero out tenors situated significantly past bond maturity
            if kt > self.maturity + 2.0 and (i > 0 and kts[i - 1] >= self.maturity):
                krds[kt] = 0.0
                continue

            # Triangular partition of unity kernel weights
            weights = np.zeros_like(self.payment_times)
            if i > 0:
                t_prev = kts[i - 1]
                left_mask = (self.payment_times >= t_prev) & (self.payment_times <= kt)
                if kt > t_prev:
                    weights[left_mask] = (self.payment_times[left_mask] - t_prev) / (kt - t_prev)
            else:
                weights[self.payment_times <= kt] = 1.0

            if i < len(kts) - 1:
                t_next = kts[i + 1]
                right_mask = (self.payment_times > kt) & (self.payment_times <= t_next)
                if t_next > kt:
                    weights[right_mask] = (t_next - self.payment_times[right_mask]) / (t_next - kt)
            else:
                weights[self.payment_times > kt] = 1.0

            # Central-difference pricing with local yield curve shift
            bump_up_df = (1.0 + (ytm + weights * dy) / m) ** (-m * self.payment_times)
            bump_dn_df = (1.0 + (ytm - weights * dy) / m) ** (-m * self.payment_times)

            p_up = float(np.sum(self.cash_flows * bump_up_df))
            p_down = float(np.sum(self.cash_flows * bump_dn_df))

            krd = (p_down - p_up) / (2.0 * dy * dirty_price)
            krds[kt] = float(max(0.0, krd))

        return BondRiskMetrics(
            clean_price=clean_price,
            dirty_price=dirty_price,
            accrued_interest=accrued_interest,
            ytm=ytm,
            macaulay_duration=mac_dur,
            modified_duration=mod_dur,
            dv01=dv01,
            convexity=convexity,
            key_rate_durations=krds,
        )


if __name__ == "__main__":
    # Test with a benchmark 10-Year Treasury Bond:
    # 4.0% coupon, trading at 98.50 dirty price
    bond = FixedRateBond(maturity=10.0, coupon_rate=0.040, face_value=100.0, frequency=2)
    metrics = bond.compute_risk_metrics(dirty_price=98.50, accrued_interest=0.50)

    print("--- BENCHMARK 10Y BOND ANALYTICS ---")
    print(f"Clean Price:         ${metrics.clean_price:.3f}")
    print(f"Dirty Price:         ${metrics.dirty_price:.3f}")
    print(f"Accrued Interest:    ${metrics.accrued_interest:.3f}")
    print(f"Yield to Maturity:   {metrics.ytm * 100:.3f}%")
    print(f"Macaulay Duration:   {metrics.macaulay_duration:.3f} years")
    print(f"Modified Duration:   {metrics.modified_duration:.3f}")
    print(f"DV01 (per $100 par): ${metrics.dv01:.4f}")
    print(f"Convexity:           {metrics.convexity:.3f}")

    print("\n--- KEY RATE DURATIONS (KRD) ---")
    total_krd = 0.0
    for tenor, krd in metrics.key_rate_durations.items():
        total_krd += krd
        print(f"  {tenor:5.1f}Y Tenor: {krd:.3f}")
    print(f"Sum of KRDs:         {total_krd:.3f} (Compare to ModDur: {metrics.modified_duration:.3f})")