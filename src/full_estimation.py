#!/usr/bin/env python3
import argparse, json, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Utilities
# ----------------------------
def model(theta_deg, M, X, t):
    """Parametric model x(t), y(t) given theta (deg), M, X and vector t."""
    th = np.deg2rad(theta_deg)
    e = np.exp(M * np.abs(t))
    s = np.sin(0.3 * t)
    x = t * np.cos(th) - e * s * np.sin(th) + X
    y = 42 + t * np.sin(th) + e * s * np.cos(th)
    return x, y

def l1_loss(theta_deg, M, X, t, x_data, y_data):
    """Assignment metric: sum of L1 distances in x and y."""
    xh, yh = model(theta_deg, M, X, t)
    return float(np.sum(np.abs(x_data - xh) + np.abs(y_data - yh)))

# ----------------------------
# Initial guesses
# ----------------------------
def estimate_theta_pca(xy):
    """Estimate theta from the principal direction of (x,y) (in degrees)."""
    Xc = xy - xy.mean(axis=0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    v1 = Vt[0]
    ang = np.degrees(np.arctan2(v1[1], v1[0]))  # [-180, 180]
    if ang < 0:
        ang += 180.0
    # Assignment bounds say 0..50 deg; clamp to sensible range without breaking init:
    return float(np.clip(ang, 0.0, 50.0))

def estimate_X_mean_align(theta_deg, x_mean, t_mean):
    """X ≈ mean_x − cos(theta)*mean_t (sin term averages ~0 over multiple cycles)."""
    return float(x_mean - math.cos(math.radians(theta_deg)) * t_mean)

def estimate_M_envelope(theta_deg, X0, xy, t):
    """
    After removing (X0, 42) and unrotating by -theta0, second coord ~ s(t)=e^{M|t|} sin(0.3 t).
    Fit: log| s / sin(0.3 t) | ≈ M |t| (skip near sin zeros).
    """
    th = math.radians(theta_deg)
    R_minus = np.array([[ math.cos(th),  math.sin(th)],
                        [-math.sin(th),  math.cos(th)]])
    latent = (xy - np.array([X0, 42.0])) @ R_minus.T
    s_hat = latent[:, 1]

    eps = 1e-8
    denom = np.sin(0.3 * t)
    mask = np.abs(denom) > 0.2  # avoid unstable divisions
    if np.count_nonzero(mask) < 10:
        return 0.0  # fallback: near-constant amplitude

    y_log = np.log(np.abs(s_hat[mask]) / (np.abs(denom[mask]) + eps) + eps)
    X_lin = np.vstack([np.abs(t[mask]), np.ones(np.count_nonzero(mask))]).T
    coef, *_ = np.linalg.lstsq(X_lin, y_log, rcond=None)
    M0 = float(coef[0])  # slope ~ M
    return float(np.clip(M0, -0.05, 0.05))

# ----------------------------
# Optimizer (Powell with fallback)
# ----------------------------
def refine_params(theta0, M0, X0, t, x_data, y_data):
    start = np.array([theta0, M0, X0], dtype=float)
    bounds = np.array([[0.0, 50.0], [-0.05, 0.05], [0.0, 100.0]], dtype=float)

    try:
        import scipy.optimize as opt

        def obj(v):
            v = np.clip(v, bounds[:,0], bounds[:,1])
            return l1_loss(v[0], v[1], v[2], t, x_data, y_data)

        res = opt.minimize(
            obj, x0=start, method="Powell",
            bounds=bounds.tolist(),
            options={"xtol": 1e-4, "ftol": 1e-4, "maxiter": 2000, "disp": False}
        )
        v = np.clip(res.x, bounds[:,0], bounds[:,1])
        return float(v[0]), float(v[1]), float(v[2]), float(res.fun), "Powell (SciPy)"
    except Exception:
        # Fallback: random-restart coordinate search near start
        rng = np.random.default_rng(0)
        best = start.copy()
        best_loss = l1_loss(*best, t, x_data, y_data)
        step = np.array([5.0, 0.01, 5.0], dtype=float)
        for _ in range(200):
            cand = best + rng.normal(scale=step, size=3)
            cand = np.clip(cand, bounds[:,0], bounds[:,1])
            loss = l1_loss(*cand, t, x_data, y_data)
            if loss < best_loss:
                best, best_loss = cand, loss
                step *= 0.9
        return float(best[0]), float(best[1]), float(best[2]), float(best_loss), "random local search"

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="Path to xy_data.csv")
    ap.add_argument("--tmin", type=float, default=6.0)
    ap.add_argument("--tmax", type=float, default=60.0)
    ap.add_argument("--savefigs", action="store_true", help="Save figures to ./figures")
    args = ap.parse_args()

    # Load data
    df = pd.read_csv(args.data)
    x_data = df["x"].to_numpy()
    y_data = df["y"].to_numpy()
    xy = df[["x", "y"]].to_numpy()
    N = len(df)
    t = np.linspace(args.tmin, args.tmax, N)

    # --- Initial estimates
    theta0 = estimate_theta_pca(xy)
    X0 = estimate_X_mean_align(theta0, x_mean=float(df["x"].mean()), t_mean=(args.tmin + args.tmax)/2)
    M0 = estimate_M_envelope(theta0, X0, xy, t)

    # --- Refine with L1 objective
    theta, M, X, L1, how = refine_params(theta0, M0, X0, t, x_data, y_data)

    # --- Print results
    print("==== Initial guesses ====")
    print(f"theta0 (deg) = {theta0:.6f}")
    print(f"M0           = {M0:.6f}")
    print(f"X0           = {X0:.6f}")
    print()
    print("==== Refined (assignment metric: L1) ====")
    print(f"theta (deg)  = {theta:.12f}")
    print(f"M            = {M:.12f}")
    print(f"X            = {X:.12f}")
    print(f"L1 distance  = {L1:.4f}")
    print(f"optimizer    = {how}")

    # --- Desmos / LaTeX string
    theta_rad = float(np.deg2rad(theta))
    latex = (
        r"\left(t\cos({th})-e^{{{M}\left|t\right|}}\cdot\sin(0.3t)\sin({th})+{X},\ "
        r"42+t\sin({th})+e^{{{M}\left|t\right|}}\cdot\sin(0.3t)\cos({th})\right)"
    ).format(th=f"{theta_rad:.6f}", M=f"{M:.6f}", X=f"{X:.6f}")
    print("\nDesmos/LaTeX (use domain 6 ≤ t ≤ 60):")
    print(latex)

    # --- Save machine-readable params
    params = dict(theta_deg=theta, theta_rad=theta_rad, M=M, X=X, L1=L1, method=how,
                  N=N, t_min=args.tmin, t_max=args.tmax)
    with open("params.json", "w") as f:
        json.dump(params, f, indent=2)

    # --- Figures
    x_fit, y_fit = model(theta, M, X, t)

    # 1) Data scatter
    plt.figure()
    plt.scatter(x_data, y_data, s=8)
    plt.title("Scatter of given points")
    plt.xlabel("x"); plt.ylabel("y")
    if args.savefigs:
        import os; os.makedirs("figures", exist_ok=True)
        plt.savefig("figures/scatter.png", dpi=160, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    # 2) Fit overlay
    plt.figure()
    plt.scatter(x_data, y_data, s=8, label="data")
    plt.plot(x_fit, y_fit, linewidth=2, label="model (fit)")
    plt.title("Curve fit to data"); plt.xlabel("x"); plt.ylabel("y"); plt.legend()
    if args.savefigs:
        plt.savefig("figures/fit_overlay.png", dpi=160, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    # 3) Envelope regression visualization (optional)
    th0 = math.radians(theta0)
    Rm = np.array([[ math.cos(th0),  math.sin(th0)],
                   [-math.sin(th0),  math.cos(th0)]])
    latent = (xy - np.array([X0, 42.0])) @ Rm.T
    s_hat = latent[:,1]
    eps = 1e-8
    denom = np.sin(0.3 * t)
    mask = np.abs(denom) > 0.2
    y_log = np.log(np.abs(s_hat[mask]) / (np.abs(denom[mask]) + eps) + eps)
    plt.figure()
    plt.scatter(np.abs(t[mask]), y_log, s=8, label="log(|s/sin(0.3t)|)")
    # regression line with M0
    c_intercept = np.linalg.lstsq(
        np.vstack([np.abs(t[mask]), np.ones(np.count_nonzero(mask))]).T, y_log, rcond=None
    )[0][1]
    plt.plot(np.abs(t[mask]), M0 * np.abs(t[mask]) + c_intercept, label=f"init slope ≈ {M0:.5f}")
    plt.title("Envelope regression to estimate M")
    plt.xlabel("|t|"); plt.ylabel("log amplitude"); plt.legend()
    if args.savefigs:
        plt.savefig("figures/envelope_fit.png", dpi=160, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    main()
