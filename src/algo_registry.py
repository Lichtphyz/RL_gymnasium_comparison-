"""
Single source of truth mapping each named "variant" in this project to its
SB3 algorithm class and its config file. Used by train.py, evaluate.py,
record_demo.py, and plot_comparison.py so they can never disagree with each
other about what a variant name means -- this is what the action_noise/PPO
bug earlier came from (four separate copies of the same dict drifting).

A ddpg_1critic variant existed at one point to force n_critics=1 on DDPG,
based on documentation suggesting SB3's DDPG defaults to n_critics=2 (via
its policy being a TD3Policy alias). Empirically checking the installed
SB3 version (2.9.0) showed regular ddpg already uses n_critics=1 by
default, making that variant redundant (confirmed by both configs
producing byte-identical trained models). Removed accordingly -- the
regular ddpg vs. td3 comparison was already the clean single-critic-vs-
twin-critic ablation this whole time.
"""

from stable_baselines3 import DDPG, PPO, SAC, TD3

# variant name -> (SB3 algorithm class, config file stem under configs/)
VARIANTS = {
    "ddpg": (DDPG, "ddpg"),
    "td3": (TD3, "td3"),
    "sac": (SAC, "sac"),
    "ppo": (PPO, "ppo"),
}

# Variants whose SB3 constructor accepts action_noise (off-policy, continuous
# action-space algorithms). PPO does not accept this argument at all.
NOISE_CAPABLE_VARIANTS = {"ddpg", "td3", "sac"}


def get_algo_class(variant: str):
    return VARIANTS[variant][0]


def get_config_name(variant: str) -> str:
    return VARIANTS[variant][1]


def variant_choices():
    return list(VARIANTS.keys())
