# Bayesian Optimization

An interactive Bayesian optimization implementation. A Gaussian Process surrogate models a black-box function from noisy observations. An acquisition function (UCB or EI) guides where to sample next to find the optimum efficiently.

I was curious to try out Bayesian optimization because it balances exploration and exploitation in a way gradient descent doesn't, and I wanted to actually conceptually understand all of the math in a way I wouldn't be able to from just watching a YouTube video. (Also, Bayesian thinking is fun to experiment with, and I didn't want to play with yet another conjugate prior/posterior.)

**[Live demo](https://cameronake.github.io/mcop-demo.html)**

All functions for the actual Bayesian optimization process are contained in `bo.py`.

---

## Functions in `bo.py`

| Function | Purpose |
|----------|---------|
| `rbf_kernel` | RBF (squared exponential) kernel matrix |
| `gp_posterior` | Cholesky-based posterior mean + std. dev. |
| `acquisition_ucb` | Upper Confidence Bound: `μ(x) + κ·σ(x)` |
| `acquisition_ei` | Expected Improvement (using `math.erfc`) |
| `maximize_acquisition` | Grid argmax over the acquisition function |
| `bo_step` | Full BO iteration; returns `{mu, sigma, acq, next_x}` |

## Stack

- Python + NumPy (Cholesky decomposition)
- [Pyodide](https://pyodide.org): Python via WebAssembly (demo runs client-side with no backend)
- Chart.js for visualization
