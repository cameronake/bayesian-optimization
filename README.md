# Bayesian Optimization

An interactive Bayesian Optimization implementation built from scratch in Python. A Gaussian Process surrogate models a black-box function from noisy observations; an acquisition function (UCB or EI) guides where to sample next to find the optimum efficiently.

**[Live demo →](https://cameronake.github.io/bo-demo.html)**
Step through the algorithm in your browser and watch the GP posterior update in real time.

---

## What's in `bo.py`

| Function | Purpose |
|----------|---------|
| `rbf_kernel` | RBF (squared exponential) kernel matrix |
| `gp_posterior` | Cholesky-based posterior mean + std (no matrix inversion) |
| `acquisition_ucb` | Upper Confidence Bound: `μ(x) + κ·σ(x)` |
| `acquisition_ei` | Expected Improvement via `math.erfc` (no scipy) |
| `maximize_acquisition` | Grid argmax over the acquisition function |
| `bo_step` | Full BO iteration; returns `{mu, sigma, acq, next_x}` |

## Stack

- Python + NumPy (Cholesky decomposition, no scipy)
- [Pyodide](https://pyodide.org) for running Python in the browser via WebAssembly
- Chart.js for visualization
