# Nelson-Siegel-Svensson term structure calibration module.

import numpy as np
from scipy.optimize import minimize


# Holds fitted Nelson-Siegel-Svensson parameters and curve functions
class NSSParameters:
    def __init__(self, beta0, beta1, beta2, beta3, lambda1, lambda2, rmse):
        self.beta0 = float(beta0)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.beta3 = float(beta3)
        self.lambda1 = float(lambda1)
        self.lambda2 = float(lambda2)
        self.rmse = float(rmse)

    # Vectorized evaluation of spot rates for maturities tau (in years)
    def zero_rate(self, tau):
        t = np.maximum(np.asarray(tau, dtype=float), 1e-6)
        term1 = (1.0 - np.exp(-t / self.lambda1)) / (t / self.lambda1)
        term2 = term1 - np.exp(-t / self.lambda1)
        term3 = ((1.0 - np.exp(-t / self.lambda2)) / (t / self.lambda2)) - np.exp(-t / self.lambda2)
        return self.beta0 + self.beta1 * term1 + self.beta2 * term2 + self.beta3 * term3

    # Evaluates theoretical instantaneous forward curve f(tau)
    def instantaneous_forward_rate(self, tau):
        t = np.maximum(np.asarray(tau, dtype=float), 1e-6)
        return (
            self.beta0
            + self.beta1 * np.exp(-t / self.lambda1)
            + self.beta2 * (t / self.lambda1) * np.exp(-t / self.lambda1)
            + self.beta3 * (t / self.lambda2) * np.exp(-t / self.lambda2)
        )


# Calibrates NSS parameters against observed market par/spot yields
class SvenssonCalibrator:
    @staticmethod
    def loss_function(params, tenors, market_yields):
        b0, b1, b2, b3, l1, l2 = params
        model = NSSParameters(b0, b1, b2, b3, l1, l2, rmse=0.0)
        
        # Scale residuals to basis points to prevent optimization gradients from vanishing
        errors = (model.zero_rate(tenors) - market_yields) * 10_000.0
        return float(np.sum(errors ** 2))

    def fit(self, tenors, market_yields, initial_guess=None):
        t = np.asarray(tenors, dtype=float)
        y = np.asarray(market_yields, dtype=float)

        # Baseline guesses: anchor beta0 to long rate and beta1 to curve spread
        if initial_guess is None:
            b0_guess = float(y[-1])
            b1_guess = float(y[0] - y[-1])
            initial_guess = np.array([b0_guess, b1_guess, 0.0, 0.0, 1.5, 5.0])

        # Financial parameter bounds to enforce asymptotic stability
        bounds = (
            (0.00, 0.20),     # beta0: long-term level
            (-0.20, 0.20),    # beta1: slope
            (-0.20, 0.20),    # beta2: curvature 1
            (-0.20, 0.20),    # beta3: curvature 2
            (0.1, 30.0),      # lambda1: decay scale 1
            (0.1, 30.0),      # lambda2: decay scale 2
        )

        result = minimize(
            fun=self.loss_function,
            x0=initial_guess,
            args=(t, y),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 2000, "ftol": 1e-9},
        )

        # Retry with wider hump seeds if L-BFGS-B gets trapped in an early local minimum
        if not result.success:
            fallback_guess = np.array([initial_guess[0], initial_guess[1], -0.01, 0.01, 2.0, 8.0])
            result = minimize(
                fun=self.loss_function,
                x0=fallback_guess,
                args=(t, y),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 3000, "ftol": 1e-9},
            )

        b0, b1, b2, b3, l1, l2 = result.x
        pred_yields = NSSParameters(b0, b1, b2, b3, l1, l2, 0.0).zero_rate(t)
        rmse_bps = float(np.sqrt(np.mean(((pred_yields - y) * 10_000.0) ** 2)))

        return NSSParameters(b0, b1, b2, b3, l1, l2, rmse=rmse_bps)


if __name__ == "__main__":
    # Test Nelson-Siegel-Svensson calibration with a sample market curve
    test_tenors = np.array([1/12, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])
    
    # Inverted yield curve snapshot (similar to early 2024 CMT)
    test_yields = np.array([0.0555, 0.0546, 0.0524, 0.0480, 0.0433, 0.0409, 0.0393, 0.0395, 0.0395, 0.0425, 0.0408])

    print("--- FITTING NELSON-SIEGEL-SVENSSON CURVE ---")
    calibrator = SvenssonCalibrator()
    params = calibrator.fit(test_tenors, test_yields)

    print("Optimal Parameters:")
    print(f"  beta0 (Long-term level):     {params.beta0 * 100:.4f}%")
    print(f"  beta1 (Short-term spread):   {params.beta1 * 100:.4f}%")
    print(f"  beta2 (Curvature 1):         {params.beta2 * 100:.4f}%")
    print(f"  beta3 (Curvature 2):         {params.beta3 * 100:.4f}%")
    print(f"  lambda1 (Scale 1):           {params.lambda1:.4f}")
    print(f"  lambda2 (Scale 2):           {params.lambda2:.4f}")
    print(f"  Fit RMSE:                    {params.rmse:.2f} bps")

    print("\n--- COMPARISON TABLE ---")
    fitted = params.zero_rate(test_tenors)
    for t, mkt, fit in zip(test_tenors, test_yields, fitted):
        diff_bps = (fit - mkt) * 10_000
        print(f"Tenor {t:5.2f}Y | Market: {mkt*100:6.3f}% | Fitted: {fit*100:6.3f}% | Error: {diff_bps:+5.2f} bps")