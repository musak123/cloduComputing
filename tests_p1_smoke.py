from training.p1.baselines import run_sequential_baseline, compare_joint_vs_baseline
from training.p1.alcaf import optimize_joint


def test_joint_beats_baseline_energy_at_85():
    base = run_sequential_baseline([0.0,0.3,0.5,0.7],["FP16","INT8","INT4"],["dense","MoE-2","MoE-4"],"LLaMA-2-7B","SQuAD")
    joint = optimize_joint([0.0,0.3,0.5,0.7],["FP16","INT8","INT4"],["dense","MoE-2","MoE-4"],"LLaMA-2-7B","SQuAD",0.7,0.3,seed=1,warmup_random_points=20,iterations=20)
    cmp = compare_joint_vs_baseline(joint, base)
    assert cmp["relative_energy_gain_at_85"] > 0.1
