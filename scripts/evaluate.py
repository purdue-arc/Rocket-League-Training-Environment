"""Evaluate a policy (blue) against an opponent (orange) over full matches with fixed seeds.

Run from the project root with the venv active:
  python scripts/evaluate.py runs/selfplay/latest.pt --opponent random
  python scripts/evaluate.py runs/selfplay/latest.pt --opponent chase --matches 64
  python scripts/evaluate.py random --opponent random          # baseline
Opponents: random, chase, idle, or another checkpoint.
"""
import argparse
from dataclasses import replace

import carSoccer.config as cfg
from carSoccer.env import makeEnv
from carSoccer.learn.play import evaluate, makeController

def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("policy", help="checkpoint .pt, or random / chase / idle")
  parser.add_argument("--opponent", default="random")
  parser.add_argument("--matches", type=int, default=32)
  parser.add_argument("--worlds", type=int, default=32)
  parser.add_argument("--seconds", type=float, default=60.0, help="match length")
  parser.add_argument("--realistic", action="store_true", help="randomised cars and physics, noisy sensors")
  parser.add_argument("--seed", type=int, default=123)
  args = parser.parse_args()

  game = replace(cfg.GameConfig(), matchSeconds=args.seconds)
  env = makeEnv(args.worlds, game=game, randomize=args.realistic, noise=args.realistic, seed=args.seed)
  result = evaluate(env, makeController(args.policy, env), makeController(args.opponent, env), args.matches)
  env.close()
  print(f"{args.policy} (blue) vs {args.opponent} (orange), {args.matches} x {args.seconds:.0f} s matches")
  print(f"  win / draw / loss   {result['win']:.0%} / {result['draw']:.0%} / {result['loss']:.0%}")
  print(f"  goals per match     for {result['goalsFor']:.2f}, against {result['goalsAgainst']:.2f}")
  print(f"  touches / car-min   blue {result['blueTouchesPerCarMin']:.2f}, orange {result['orangeTouchesPerCarMin']:.2f}")

if __name__ == "__main__":
  main()
