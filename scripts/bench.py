"""Env steps per second: physics only, the full engine step, and the env with observations.

Run from the project root with the venv active:
  python scripts/bench.py --worlds 256 --threads 1 4 9
One env step is one control tick of every world (20 substeps at 1 ms); numbers are summed over worlds.
"""
import argparse, time

import torch

import carSoccer.config as cfg
from carSoccer.components.bots import RandomDriver
from carSoccer.engine.sim.backend.cpuRollout import CpuRolloutBackend
from carSoccer.engine.sim.scene.builder import buildScene
from carSoccer.env import makeEnv

def rate(fn, worlds, steps):
  fn()                                     # warm up
  start = time.perf_counter()
  for _ in range(steps):
    fn()
  return worlds * steps / (time.perf_counter() - start)

def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--worlds", type=int, default=256)
  parser.add_argument("--steps", type=int, default=30, help="env steps (control ticks) to time")
  parser.add_argument("--threads", type=int, nargs="+", default=[1, 4, 9])
  parser.add_argument("--realistic", action="store_true", help="per-match randomisation and sensor noise in the env")
  args = parser.parse_args()

  scene, sceneCfg, game = buildScene(cfg.SceneConfig()), cfg.SceneConfig(), cfg.GameConfig()
  nsub = round(1 / game.controlHz / sceneCfg.sim.timestep)
  print(f"{args.worlds} worlds, {nsub} substeps per env step, torch threads {torch.get_num_threads()}")
  print(f"{'nthread':>7} | {'physics':>10} | {'engine':>10} | {'env+obs':>10} | engine overhead")
  for nthread in args.threads:
    backend = CpuRolloutBackend(scene.model, args.worlds, nthread)
    ctrl = torch.zeros(args.worlds, scene.model.nu, dtype=torch.float64)
    physics = rate(lambda: backend.step(ctrl, None, nsub), args.worlds, args.steps)
    backend.close()

    env = makeEnv(args.worlds, nthread=nthread, randomize=args.realistic, noise=args.realistic)
    env.reset()
    driver = RandomDriver(args.worlds, env.ctx.carsPerTeam, env.ctx)
    engine = rate(lambda: env.engine.step(driver(env.state)), args.worlds, args.steps)
    decisions = max(1, args.steps // env.actionRepeat)
    full = rate(lambda: env.step(driver(env.state)), args.worlds * env.actionRepeat, decisions)
    env.close()
    print(f"{nthread:>7} | {physics:>10,.0f} | {engine:>10,.0f} | {full:>10,.0f} | {physics / engine - 1:+.1%}")

if __name__ == "__main__":
  main()
