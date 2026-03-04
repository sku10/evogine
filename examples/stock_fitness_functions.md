# Stock Backtesting Fitness Functions

When optimizing trading strategies with evogine, the fitness function determines what
"better" means. The choice of ratio shapes what the GA optimizes for — and getting this
right matters as much as the optimizer settings.

---

## Common Risk-Adjusted Return Ratios

| Ratio | Formula | Best when |
|---|---|---|
| **Calmar** | Annualized return / max drawdown | You care about catastrophic loss above all |
| **Sharpe** | (Return − risk-free) / std dev of returns | Standard benchmark; penalizes all volatility |
| **Sortino** | (Return − risk-free) / downside std dev | Only penalizes bad volatility, not upside swings |
| **MAR** | Total return / max drawdown | Like Calmar but over full history, not annualized |
| **Omega** | Area above threshold / area below | Distribution-free; works with non-normal returns |

---

## As evogine Fitness Functions

```python
def calmar(ind):
    """Annualized return per unit of max drawdown. Good default for trend strategies."""
    r = backtest(ind)
    if r['max_drawdown'] == 0:
        return 0.0
    return r['annualized_return'] / r['max_drawdown']


def sortino(ind):
    """Like Sharpe but only penalizes downside volatility — rewards big up moves."""
    r = backtest(ind)
    if r['downside_std'] == 0:
        return 0.0
    return (r['annual_return'] - 0.02) / r['downside_std']  # 0.02 = risk-free rate


def sharpe(ind):
    """Classic risk-adjusted return. Penalizes all volatility, up and down."""
    r = backtest(ind)
    if r['return_std'] == 0:
        return 0.0
    return (r['annual_return'] - 0.02) / r['return_std']


def mar(ind):
    """Total return over full history divided by max drawdown. No annualization."""
    r = backtest(ind)
    if r['max_drawdown'] == 0:
        return 0.0
    return r['total_return'] / r['max_drawdown']
```

---

## Choosing the Right Ratio

**Calmar / MAR** — use when your strategy trades infrequently and drawdown duration
matters. A 50% drawdown that lasts 2 years is worse than one that recovers in 2 weeks;
the Calmar doesn't capture that, but it's still the most widely used in hedge fund
evaluation.

**Sharpe** — use when you want a well-understood, industry-standard metric you can
compare to benchmarks (S&P 500 Sharpe is typically ~0.5–0.7). It penalizes upside
volatility too, which can hurt momentum strategies unfairly.

**Sortino** — use when your strategy naturally has asymmetric returns (e.g. options,
momentum). Only downside deviation is penalized, so large winning streaks don't count
against you.

**Omega** — use when returns are non-normal (fat tails, skew). More computationally
expensive but the most complete picture.

---

## Multi-Objective: Let the Pareto Front Decide

Rather than committing to a single ratio upfront, optimize all objectives simultaneously
and inspect the trade-off frontier afterward:

```python
from evogine import MultiObjectiveGA

def objectives(ind):
    r = backtest(ind)
    return (
        r['annualized_return'],   # maximize
        -r['max_drawdown'],       # minimize (negate so higher = better internally)
        r['win_rate'],            # maximize
    )

mo = MultiObjectiveGA(
    gene_builder     = genes,
    fitness_function = objectives,
    n_objectives     = 3,
    objectives       = ['maximize', 'maximize', 'maximize'],  # all already oriented
    population_size  = 100,
    generations      = 200,
    seed             = 42,
)

pareto_front, history = mo.run()

# Inspect the frontier — pick your preferred return/drawdown trade-off
for sol in pareto_front:
    ret, dd, wr = sol['scores']
    print(f"Return={ret:.1%}  Drawdown={-dd:.1%}  WinRate={wr:.1%}  Params={sol['individual']}")
```

This gives you all the non-dominated strategies at once. You can see what maximum return
is achievable at a given drawdown limit, rather than baking that preference into a ratio
before you know what's possible.

---

## Guarding Against Degenerate Results

The GA will exploit any weakness in the fitness function. Common gotchas:

```python
def robust_calmar(ind):
    r = backtest(ind)

    # Minimum trade count — avoid strategies that barely trade
    if r['n_trades'] < 10:
        return -999.0

    # Minimum time in market — avoid "never enter" degenerate solutions
    if r['time_in_market'] < 0.05:
        return -999.0

    # Guard zero drawdown (can mean no trades, not a perfect strategy)
    if r['max_drawdown'] == 0:
        return 0.0

    return r['annualized_return'] / r['max_drawdown']
```

Or use evogine's built-in constraint system to enforce these cleanly:

```python
ga = GeneticAlgorithm(
    genes, calmar,
    constraints=[
        lambda ind: backtest(ind)['n_trades'] >= 10,
        lambda ind: backtest(ind)['time_in_market'] >= 0.05,
    ],
)
```

Note: constraints re-run the backtest, so cache results if your backtest is slow.

---

## Related

- `examples/stock_strategy_optimization.md` — full end-to-end setup, backtest engines,
  overfitting prevention, walk-forward optimization
