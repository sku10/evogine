"""
Live evolution plot using the on_generation callback.

Shows best score and average score updating in real-time as the GA runs.
Optimizes the Rastrigin function — a classic benchmark with many local optima.

Run:
    python examples/live_plot.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from genetic_engine import (
    GeneticAlgorithm,
    GeneBuilder,
    FloatRange,
    TournamentSelection,
    ArithmeticCrossover,
)


# ---------------------------------------------------------------------------
# Problem: Rastrigin function (2D)
# Many local optima, global optimum at (0, 0) with score 0
# ---------------------------------------------------------------------------

def rastrigin(ind: dict) -> float:
    x, y = ind['x'], ind['y']
    n = 2
    result = 10 * n + (x**2 - 10 * math.cos(2 * math.pi * x)) \
                    + (y**2 - 10 * math.cos(2 * math.pi * y))
    return -result  # negate: we maximize, Rastrigin is minimized at 0


# ---------------------------------------------------------------------------
# Live plot setup
# ---------------------------------------------------------------------------

fig = plt.figure(figsize=(12, 5))
fig.suptitle("Genetic Algorithm — Live Evolution (Rastrigin 2D)", fontsize=13)
gs = gridspec.GridSpec(1, 2, figure=fig)

ax_score = fig.add_subplot(gs[0])
ax_score.set_title("Fitness over generations")
ax_score.set_xlabel("Generation")
ax_score.set_ylabel("Score (higher = better)")
ax_score.set_xlim(0, 1)
ax_score.set_ylim(-80, 5)
ax_score.grid(True, alpha=0.3)

line_best, = ax_score.plot([], [], 'b-', linewidth=2, label='Best score')
line_avg,  = ax_score.plot([], [], 'r--', linewidth=1, label='Avg score')
ax_score.legend()

ax_space = fig.add_subplot(gs[1])
ax_space.set_title("Best individual position (x, y)")
ax_space.set_xlabel("x")
ax_space.set_ylabel("y")
ax_space.set_xlim(-5.12, 5.12)
ax_space.set_ylim(-5.12, 5.12)
ax_space.axhline(0, color='gray', linewidth=0.5)
ax_score.axvline(0, color='gray', linewidth=0.5) if False else None
ax_space.plot(0, 0, 'g*', markersize=15, label='Optimum (0,0)')
best_dot, = ax_space.plot([], [], 'ro', markersize=10, label='Current best')
ax_space.legend()
ax_space.grid(True, alpha=0.3)

plt.tight_layout()
plt.pause(0.1)

gens_history = []
best_history = []
avg_history  = []

TOTAL_GENS = 150


def on_generation(gen, best_score, avg_score, best_individual):
    gens_history.append(gen)
    best_history.append(best_score)
    avg_history.append(avg_score)

    # Update score lines
    line_best.set_data(gens_history, best_history)
    line_avg.set_data(gens_history, avg_history)
    ax_score.set_xlim(0, max(TOTAL_GENS, gen + 1))
    ax_score.set_ylim(min(avg_history) * 1.1, 5)

    # Update position dot
    if best_individual is not None:
        best_dot.set_data([best_individual['x']], [best_individual['y']])

    fig.canvas.draw()
    plt.pause(0.01)  # yields control to matplotlib to redraw


# ---------------------------------------------------------------------------
# Run the GA
# ---------------------------------------------------------------------------

genes = GeneBuilder()
genes.add('x', FloatRange(-5.12, 5.12, sigma=0.15))
genes.add('y', FloatRange(-5.12, 5.12, sigma=0.15))

ga = GeneticAlgorithm(
    gene_builder=genes,
    fitness_function=rastrigin,
    population_size=80,
    generations=TOTAL_GENS,
    mutation_rate=0.3,
    crossover_rate=0.7,
    elitism=2,
    seed=42,
    patience=30,
    selection=TournamentSelection(k=4),
    crossover=ArithmeticCrossover(),
    on_generation=on_generation,
)

best, score, history = ga.run()

print(f"\nBest individual: x={best['x']:.4f}, y={best['y']:.4f}")
print(f"Best score: {score:.6f}  (optimum is 0.0)")
print(f"Generations run: {len(history)}")

ax_score.set_title(f"Done — best score: {score:.4f}  (optimum: 0.0)")
plt.show()
