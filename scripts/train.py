"""Train a shared-weight 3v3 self-play policy with PPO on the gameplay reward (touches, goals).

Run from the project root with the venv active:
  python scripts/train.py                               # defaults: 256 worlds, 300 iterations, runs/selfplay
  python scripts/train.py --name quick --worlds 64 --iterations 50
  python scripts/train.py --resume runs/selfplay/latest.pt --iterations 600
Then watch it play:
  mjpython scripts/watchMatch.py --policy runs/selfplay/latest.pt
"""
import argparse
from dataclasses import replace

import torch

from carSoccer.learn.runner import Learner, TrainConfig, configFromDict

def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--name", default=None)
  parser.add_argument("--worlds", type=int, default=None)
  parser.add_argument("--steps", type=int, default=None, help="decisions per world per rollout")
  parser.add_argument("--iterations", type=int, default=None)
  parser.add_argument("--threads", type=int, default=None, help="MuJoCo rollout threads")
  parser.add_argument("--seed", type=int, default=None)
  parser.add_argument("--no-noise", action="store_true", help="perfect sensors")
  parser.add_argument("--no-randomize", action="store_true", help="nominal car and physics every match")
  parser.add_argument("--resume", default=None, help="checkpoint to continue from (its config is reused)")
  args = parser.parse_args()

  train = TrainConfig()
  if args.resume:
    train = configFromDict(torch.load(args.resume, map_location="cpu", weights_only=False)["config"])
  overrides = {"name": args.name, "numWorlds": args.worlds, "steps": args.steps, "iterations": args.iterations,
               "nthread": args.threads, "seed": args.seed}
  train = replace(train, **{k: v for k, v in overrides.items() if v is not None})
  if args.no_noise:
    train = replace(train, noise=False)
  if args.no_randomize:
    train = replace(train, randomize=False)

  print(train)
  Learner(train, resume=args.resume).learn()

if __name__ == "__main__":
  main()
