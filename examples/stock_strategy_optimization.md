# Stock Trading Strategy Optimization with evogine

evogine was originally built as part of an AI stocks system, making this one of its
most natural use cases. The idea: **genes = strategy parameters, fitness function = backtest result.**

## Core Concept

```python
from evogine import GeneticAlgorithm, FloatRange, IntRange, ChoiceList, GeneBuilder

gb = GeneBuilder()
gb.add('fast_ma',       IntRange(5, 50))           # fast moving average period
gb.add('slow_ma',       IntRange(20, 200))          # slow moving average period
gb.add('rsi_threshold', FloatRange(20.0, 40.0))    # RSI oversold level
gb.add('stop_loss',     FloatRange(0.01, 0.10))    # 1–10% stop loss
gb.add('position_size', FloatRange(0.05, 0.50))   # 5–50% of capital per trade
gb.add('indicator',     ChoiceList(['sma', 'ema', 'wma']))

def fitness(params):
    # run your backtest engine with these params
    result = backtest(
        ticker='SPY',
        fast=params['fast_ma'],
        slow=params['slow_ma'],
        rsi_threshold=params['rsi_threshold'],
        stop_loss=params['stop_loss'],
        position_size=params['position_size'],
        indicator=params['indicator'],
    )
    return result['sharpe_ratio']  # or total_return, calmar_ratio, etc.

ga = GeneticAlgorithm(gb, fitness, population_size=50, generations=100, seed=42)
best_params, best_sharpe, history = ga.run()
```

## Which Optimizer for Which Problem

| Scenario | Optimizer |
|---|---|
| All numeric params (periods, thresholds) | `DEOptimizer` — best for continuous float spaces |
| Mixed (categorical indicator choice + numeric) | `GeneticAlgorithm` — handles mixed types |
| Smooth surface (few params, convex-ish) | `CMAESOptimizer` — fastest convergence |
| Suspecting many local optima | `IslandModel` — parallel exploration |
| Want to explore ALL trade-offs at once | `MultiObjectiveGA` — Sharpe vs drawdown vs win rate |

## Multi-Objective: the Pareto Front of Strategies

Rather than collapsing everything into one number, multi-objective returns a *set* of
non-dominated strategies — you pick the trade-off you prefer.

```python
from evogine import MultiObjectiveGA

def objectives(params):
    r = backtest(params)
    return (
        r['sharpe_ratio'],    # maximize
        -r['max_drawdown'],   # minimize drawdown (negate so higher = better)
        r['win_rate'],        # maximize
    )

mo = MultiObjectiveGA(gb, objectives, n_objectives=3,
                      population_size=100, generations=200)
pareto_front, history = mo.run()
# Returns a SET of strategies — pick the trade-off you want
```

## The Critical Caveat: Overfitting

The GA will find params that are perfectly optimized for the training data and perform
terribly live. Minimum viable protection:

```python
TRAIN_DATA = prices['2015':'2020']
TEST_DATA  = prices['2021':'2023']

def fitness(params):
    # Only backtest on training period
    result = backtest(params, data=TRAIN_DATA)

    # Penalize invalid configurations via constraints
    if params['fast_ma'] >= params['slow_ma']:
        return -999  # crossover system requires fast < slow

    return result['sharpe_ratio']

best_params, _, _ = ga.run()

# Validate on unseen data
test_result = backtest(best_params, data=TEST_DATA)
print(f"Out-of-sample Sharpe: {test_result['sharpe_ratio']}")
```

Even better: walk-forward optimization — re-optimize on a rolling training window
and test on the next period, repeating across the full history.

A cleaner way to enforce the fast < slow constraint using evogine's built-in constraint system:

```python
ga = GeneticAlgorithm(
    gb, fitness,
    constraints=[lambda p: p['fast_ma'] < p['slow_ma']],
    population_size=50, generations=100,
)
```

## Backtest Engine Options

evogine is engine-agnostic — the fitness function just needs to return a float.

- **[vectorbt](https://vectorbt.dev/)** — fastest, numpy-based, great for GA (thousands of backtests/sec)
- **[backtrader](https://www.backtrader.com/)** — more feature-complete, slower
- **[bt](https://pmorissette.github.io/bt/)** — simpler API
- **Custom** — even a 30-line pandas backtest works

The optimizer never sees inside `fitness()` — it calls it with a param dict and expects a float back.

## Using landscape_analysis to Pick the Right Optimizer

Before running a long optimization, sample the fitness landscape to get a recommendation:

```python
from evogine import landscape_analysis

report = landscape_analysis(gb, fitness, n_samples=200, seed=42)
print(report['recommendation'])  # e.g. 'DEOptimizer' or 'IslandModel'
print(report['reason'])
print(f"Ruggedness: {report['ruggedness']:.3f}, Modes: {report['estimated_modes']}")
```
