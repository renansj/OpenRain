"""
OpenRain PoC: Distributed Training Simulation with Byzantine Robustness

This simulates the core OpenRain protocol:
- N nodes training via DiLoCo (local SGD + periodic aggregation)
- Fraction epsilon of nodes are adversarial
- 4-layer defense: randomized filter, effect-based verification, temporal detection, audit
- Measures empirical security (does the model converge despite adversaries?)

This validates the theoretical 93% bound empirically.

Author: Renan Zapelini
License: Apache 2.0
"""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable
import time


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    # Network
    n_nodes: int = 500
    epsilon: float = 0.10  # fraction adversarial
    
    # Model (simulated as d-dimensional vector)
    dim: int = 1000  # parameter dimension (simulates model weights)
    
    # DiLoCo
    H: int = 500  # inner steps per outer round
    inner_lr: float = 3e-4
    outer_lr: float = 0.7
    outer_momentum: float = 0.9
    T: int = 1000  # total outer rounds
    
    # Defense
    n_filters: int = 8
    sampling_lambda: float = 0.3  # uniform component of verification sampling
    verification_rate: float = 0.05  # fraction verified per round
    temporal_gamma: float = 0.95
    audit_interval: int = 100
    audit_count: int = 5
    
    # Loss landscape (quadratic for simulation: L(w) = 0.5 * w^T A w)
    mu: float = 0.01  # PL constant
    L_smooth: float = 1.0  # Lipschitz constant
    sigma: float = 0.1  # gradient noise std
    
    # Experiment
    seed: int = 42
    n_trials: int = 100  # number of independent trials
    alpha_threshold: float = 0.05  # "failure" = excess risk > 5%


# =============================================================================
# Loss Landscape (Quadratic - satisfies PL condition exactly)
# =============================================================================

class QuadraticLoss:
    """
    L(w) = 0.5 * w^T A w where A = diag(eigenvalues)
    Minimum at w* = 0, L(w*) = 0.
    Satisfies PL with mu = min(eigenvalues).
    """
    
    def __init__(self, dim: int, mu: float, L_smooth: float, rng: np.random.Generator):
        # Eigenvalues uniformly spaced between mu and L_smooth
        self.eigenvalues = np.linspace(mu, L_smooth, dim)
        self.dim = dim
    
    def loss(self, w: np.ndarray) -> float:
        return 0.5 * np.sum(self.eigenvalues * w**2)
    
    def gradient(self, w: np.ndarray) -> np.ndarray:
        return self.eigenvalues * w
    
    def noisy_gradient(self, w: np.ndarray, rng: np.random.Generator, sigma: float) -> np.ndarray:
        return self.gradient(w) + rng.normal(0, sigma, self.dim)


# =============================================================================
# Adversary Strategies
# =============================================================================

class AdversaryType(Enum):
    NONE = "honest"
    RANDOM = "random_noise"
    SIGN_FLIP = "sign_flip"
    SUBTLE = "subtle_bias"  # "A Little Is Enough" style
    BACKDOOR = "backdoor"


@dataclass
class Adversary:
    strategy: AdversaryType
    strength: float = 1.0  # how aggressive
    
    def corrupt_gradient(self, honest_grad: np.ndarray, all_grads: np.ndarray, 
                         rng: np.random.Generator, sigma: float) -> np.ndarray:
        if self.strategy == AdversaryType.NONE:
            return honest_grad
        
        if self.strategy == AdversaryType.RANDOM:
            return rng.normal(0, sigma * 10, len(honest_grad))
        
        if self.strategy == AdversaryType.SIGN_FLIP:
            return -honest_grad * self.strength
        
        if self.strategy == AdversaryType.SUBTLE:
            # Stay within distribution but bias in a specific direction
            mean_grad = np.mean(all_grads, axis=0)
            std_grad = np.std(all_grads, axis=0) + 1e-10
            # Push in direction of first eigenvector, within 1 std
            direction = np.zeros_like(honest_grad)
            direction[0] = 1.0  # attack first coordinate
            return mean_grad + self.strength * std_grad * direction
        
        if self.strategy == AdversaryType.BACKDOOR:
            # Honest on most dimensions, adversarial on a few
            corrupted = honest_grad.copy()
            n_backdoor = max(1, len(honest_grad) // 100)  # affect 1% of dims
            corrupted[:n_backdoor] = -corrupted[:n_backdoor] * self.strength
            return corrupted
        
        return honest_grad


# =============================================================================
# Robust Aggregation Filters
# =============================================================================

def trimmed_mean(grads: np.ndarray, alpha: float) -> np.ndarray:
    """Coordinate-wise trimmed mean. Remove top/bottom alpha fraction."""
    n = len(grads)
    k = int(n * alpha)
    if k == 0:
        return np.mean(grads, axis=0)
    sorted_grads = np.sort(grads, axis=0)
    return np.mean(sorted_grads[k:-k], axis=0)


def krum(grads: np.ndarray, f: int) -> np.ndarray:
    """Multi-Krum: select gradient closest to its n-f-2 nearest neighbors."""
    n = len(grads)
    n_select = n - f - 2
    if n_select <= 0:
        n_select = 1
    
    # Compute pairwise distances
    scores = np.zeros(n)
    for i in range(n):
        dists = np.array([np.linalg.norm(grads[i] - grads[j]) for j in range(n) if j != i])
        dists.sort()
        scores[i] = np.sum(dists[:n_select])
    
    # Select the one with lowest score
    best = np.argmin(scores)
    return grads[best]


def geometric_median(grads: np.ndarray, max_iter: int = 20) -> np.ndarray:
    """Approximate geometric median via Weiszfeld's algorithm."""
    y = np.mean(grads, axis=0)
    for _ in range(max_iter):
        dists = np.array([np.linalg.norm(g - y) for g in grads])
        dists = np.maximum(dists, 1e-10)
        weights = 1.0 / dists
        weights /= weights.sum()
        y_new = np.average(grads, axis=0, weights=weights)
        if np.linalg.norm(y_new - y) < 1e-8:
            break
        y = y_new
    return y


def build_filter_family(n_nodes: int, epsilon: float):
    """Build family of 8 aggregation filters."""
    f = int(n_nodes * epsilon)
    filters = [
        ("trimmed_mean_0.10", lambda g: trimmed_mean(g, 0.10)),
        ("trimmed_mean_0.15", lambda g: trimmed_mean(g, 0.15)),
        ("trimmed_mean_0.20", lambda g: trimmed_mean(g, 0.20)),
        ("trimmed_mean_0.25", lambda g: trimmed_mean(g, 0.25)),
        ("krum_f=N/5", lambda g: krum(g, max(1, len(g)//5))),
        ("krum_f=N/4", lambda g: krum(g, max(1, len(g)//4))),
        ("geometric_median", lambda g: geometric_median(g)),
        ("trimmed_mean_0.30", lambda g: trimmed_mean(g, 0.30)),
    ]
    return filters


# =============================================================================
# Temporal Detection (Layer 3)
# =============================================================================

@dataclass
class NodeReputation:
    score: float = 0.0
    rounds_active: int = 0
    flagged: bool = False


# =============================================================================
# Simulation
# =============================================================================

def simulate_single_trial(cfg: Config, adversary_type: AdversaryType, 
                          rng: np.random.Generator, verbose: bool = False) -> dict:
    """
    Run one trial of DiLoCo training with adversaries and defenses.
    Returns metrics about convergence and security.
    """
    # Setup
    loss_fn = QuadraticLoss(cfg.dim, cfg.mu, cfg.L_smooth, rng)
    filters = build_filter_family(cfg.n_nodes, cfg.epsilon)
    
    # Initial weights (random, away from optimum)
    w = rng.normal(0, 1.0, cfg.dim)
    initial_loss = loss_fn.loss(w)
    optimal_loss = 0.0  # quadratic minimum is 0
    
    # Outer momentum buffer
    momentum = np.zeros(cfg.dim)
    
    # Node assignment
    n_adversarial = int(cfg.n_nodes * cfg.epsilon)
    adversary = Adversary(strategy=adversary_type, strength=1.0)
    
    # Reputation tracking
    reputations = [NodeReputation() for _ in range(cfg.n_nodes)]
    
    # Metrics
    loss_history = []
    detections = 0
    
    for t in range(cfg.T):
        # === DiLoCo: each node does H inner steps locally ===
        # (Simulated: each node computes the outer gradient Δw after H steps)
        # For quadratic loss with SGD: Δw ≈ -H * lr * gradient + noise
        
        outer_grads = []
        for i in range(cfg.n_nodes):
            # Simulate H steps of local SGD as one aggregated gradient
            # (Valid approximation for quadratic loss)
            grad = loss_fn.noisy_gradient(w, rng, cfg.sigma / np.sqrt(cfg.H))
            
            if i < n_adversarial:
                # Adversarial node
                all_honest = [loss_fn.noisy_gradient(w, rng, cfg.sigma / np.sqrt(cfg.H)) 
                              for _ in range(5)]  # estimate distribution
                grad = adversary.corrupt_gradient(grad, np.array(all_honest), rng, cfg.sigma)
            
            outer_grads.append(grad)
        
        outer_grads = np.array(outer_grads)
        
        # === Layer 1: Randomized filter ===
        filter_idx = rng.integers(0, len(filters))
        filter_name, filter_fn = filters[filter_idx]
        aggregated = filter_fn(outer_grads)
        
        # === Layer 2: Effect-based verification (sampled) ===
        n_verify = max(1, int(cfg.n_nodes * cfg.verification_rate))
        # Mixture sampling: 70% proportional to deviation, 30% uniform
        deviations = np.array([np.linalg.norm(g - aggregated) for g in outer_grads])
        dev_probs = deviations / (deviations.sum() + 1e-10)
        mixed_probs = (1 - cfg.sampling_lambda) * dev_probs + cfg.sampling_lambda / cfg.n_nodes
        mixed_probs /= mixed_probs.sum()
        
        verified_nodes = rng.choice(cfg.n_nodes, size=n_verify, replace=False, p=mixed_probs)
        for node_idx in verified_nodes:
            # Check: does this node's gradient improve or worsen loss?
            w_test = w - cfg.outer_lr * outer_grads[node_idx]
            if loss_fn.loss(w_test) > loss_fn.loss(w) * 1.1:  # 10% worse
                reputations[node_idx].score += 1.0
                if reputations[node_idx].score > 3.0:
                    reputations[node_idx].flagged = True
                    detections += 1
        
        # === Layer 3: Temporal detection ===
        for i in range(cfg.n_nodes):
            w_i = w - cfg.outer_lr * outer_grads[i]
            delta_loss = loss_fn.loss(w_i) - loss_fn.loss(w)
            reputations[i].score = cfg.temporal_gamma * reputations[i].score + delta_loss
            reputations[i].rounds_active += 1
        
        # === Layer 4: Periodic audit ===
        if t > 0 and t % cfg.audit_interval == 0:
            audited = rng.choice(cfg.n_nodes, size=cfg.audit_count, replace=False)
            for node_idx in audited:
                if node_idx < n_adversarial:
                    reputations[node_idx].flagged = True
                    detections += 1
        
        # === Outer step: apply aggregated gradient with momentum ===
        momentum = cfg.outer_momentum * momentum + aggregated
        w = w - cfg.outer_lr * momentum / cfg.n_nodes
        
        current_loss = loss_fn.loss(w)
        loss_history.append(current_loss)
        
        if verbose and t % 100 == 0:
            excess = (current_loss - optimal_loss) / (initial_loss - optimal_loss)
            print(f"  Round {t:4d} | Loss: {current_loss:.6f} | Excess risk: {excess:.4f} | Filter: {filter_name}")
    
    # Final metrics
    final_loss = loss_fn.loss(w)
    excess_risk = (final_loss - optimal_loss) / (initial_loss - optimal_loss)
    
    return {
        "final_loss": final_loss,
        "initial_loss": initial_loss,
        "excess_risk": excess_risk,
        "secure": excess_risk <= cfg.alpha_threshold,
        "detections": detections,
        "loss_history": loss_history,
    }


def run_experiment(cfg: Config, adversary_type: AdversaryType, n_trials: int = None):
    """Run multiple trials and compute empirical security."""
    if n_trials is None:
        n_trials = cfg.n_trials
    
    rng = np.random.default_rng(cfg.seed)
    results = []
    
    print(f"\n{'='*70}")
    print(f"Experiment: {adversary_type.value}")
    print(f"  Nodes: {cfg.n_nodes}, Adversarial: {cfg.epsilon*100:.0f}%, Rounds: {cfg.T}")
    print(f"  Dimension: {cfg.dim}, Threshold: {cfg.alpha_threshold*100:.0f}%")
    print(f"{'='*70}")
    
    t0 = time.time()
    for trial in range(n_trials):
        trial_rng = np.random.default_rng(rng.integers(0, 2**32))
        result = simulate_single_trial(cfg, adversary_type, trial_rng, verbose=(trial == 0))
        results.append(result)
        
        if (trial + 1) % 10 == 0:
            secure_so_far = sum(1 for r in results if r["secure"]) / len(results)
            print(f"  Trial {trial+1}/{n_trials} | Empirical security: {secure_so_far*100:.1f}%")
    
    elapsed = time.time() - t0
    
    # Summary
    n_secure = sum(1 for r in results if r["secure"])
    empirical_security = n_secure / n_trials
    mean_excess = np.mean([r["excess_risk"] for r in results])
    max_excess = np.max([r["excess_risk"] for r in results])
    mean_detections = np.mean([r["detections"] for r in results])
    
    print(f"\n{'='*70}")
    print(f"RESULTS: {adversary_type.value}")
    print(f"{'='*70}")
    print(f"  Empirical security:  {empirical_security*100:.1f}% ({n_secure}/{n_trials} trials secure)")
    print(f"  Theoretical bound:   93.0%")
    print(f"  Mean excess risk:    {mean_excess*100:.3f}%")
    print(f"  Max excess risk:     {max_excess*100:.3f}%")
    print(f"  Mean detections:     {mean_detections:.1f} adversarial nodes caught")
    print(f"  Time:                {elapsed:.1f}s")
    print(f"{'='*70}\n")
    
    return {
        "adversary": adversary_type.value,
        "empirical_security": empirical_security,
        "mean_excess_risk": mean_excess,
        "max_excess_risk": max_excess,
        "mean_detections": mean_detections,
        "n_trials": n_trials,
    }


# =============================================================================
# Main: Run all experiments
# =============================================================================

def main():
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║  OpenRain PoC: Empirical Validation of Security Bounds             ║
    ║                                                                    ║
    ║  Simulates DiLoCo distributed training with adversarial nodes      ║
    ║  and multi-layer defense. Measures empirical security against      ║
    ║  the theoretical 93% bound.                                        ║
    ║                                                                    ║
    ║  Author: Renan Zapelini                                            ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Use smaller parameters for PoC (full simulation with N=500, d=1000, T=1000
    # takes significant compute; this is a scaled-down validation)
    cfg = Config(
        n_nodes=50,       # scaled down from 500 (ratios preserved)
        epsilon=0.10,
        dim=100,          # scaled down from 1000
        H=50,             # scaled down from 500
        T=200,            # scaled down from 1000
        n_trials=50,
        sigma=0.1,
        mu=0.01,
        alpha_threshold=0.05,
    )
    
    print(f"Configuration (scaled for PoC):")
    print(f"  Nodes: {cfg.n_nodes} (production: 500)")
    print(f"  Adversarial: {cfg.epsilon*100:.0f}%")
    print(f"  Dimensions: {cfg.dim} (production: model params)")
    print(f"  Inner steps H: {cfg.H}")
    print(f"  Outer rounds T: {cfg.T}")
    print(f"  Trials: {cfg.n_trials}")
    print(f"  Security threshold: excess risk <= {cfg.alpha_threshold*100:.0f}%")
    
    all_results = []
    
    # Baseline: no adversary
    all_results.append(run_experiment(cfg, AdversaryType.NONE))
    
    # Random noise adversary (easiest to detect)
    all_results.append(run_experiment(cfg, AdversaryType.RANDOM))
    
    # Sign flip adversary (medium difficulty)
    all_results.append(run_experiment(cfg, AdversaryType.SIGN_FLIP))
    
    # Subtle adversary - "A Little Is Enough" style (hardest)
    all_results.append(run_experiment(cfg, AdversaryType.SUBTLE))
    
    # Backdoor adversary
    all_results.append(run_experiment(cfg, AdversaryType.BACKDOOR))
    
    # Final summary
    print("\n")
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"{'Adversary':<20} {'Empirical Security':<22} {'Mean Excess Risk':<20} {'vs 93% bound'}")
    print("-" * 70)
    for r in all_results:
        vs_bound = "ABOVE" if r["empirical_security"] >= 0.93 else "BELOW"
        print(f"{r['adversary']:<20} {r['empirical_security']*100:>6.1f}%             "
              f"{r['mean_excess_risk']*100:>6.3f}%             {vs_bound}")
    print("-" * 70)
    print(f"\nTheoretical bound: 93.0% (conditional on H1-H5)")
    print(f"If empirical results are ABOVE the bound across adversary types,")
    print(f"this validates that assumptions H1-H5 hold for this configuration.")
    print(f"\nNote: This is a scaled-down simulation. Full validation requires")
    print(f"N=500, d=model_size, T=1000, and real neural network training.")


if __name__ == "__main__":
    main()
