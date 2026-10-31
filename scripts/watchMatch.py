"""Watch a 3v3 match in the MuJoCo viewer.

Run from the project root with the venv active. On macOS the viewer needs mjpython:
  mjpython scripts/watchMatch.py                    # random drivers, standard kickoffs
  mjpython scripts/watchMatch.py --policy chase     # scripted ball-chasing bots
  mjpython scripts/watchMatch.py --scatter 1.0      # random kickoff positions
  mjpython scripts/watchMatch.py --randomize --noise   # random car variant and physics per match, noisy sensors
  mjpython scripts/watchMatch.py --policy runs/selfplay/latest.pt                    # a trained policy, both teams
  mjpython scripts/watchMatch.py --policy runs/selfplay/latest.pt --opponent chase   # trained blue vs scripted orange
On Linux / Windows plain `python` works.

The viewer shows world 0. It steps one control tick per frame (action repeat 1) so motion is smooth at 50 Hz;
training uses the config's action repeat. Each team's reward is printed at every goal and at full time.
"""
import argparse, os, sys, time
from dataclasses import replace

import mujoco, mujoco.viewer

def relaunchUnderMjpython() -> None:
  """macOS's viewer only runs under mjpython; if started with plain python (e.g. the editor's Run button), re-exec."""
  if sys.platform != "darwin" or mujoco.viewer._MJPYTHON is not None:
    return
  mjpython = os.path.join(os.path.dirname(sys.executable), "mjpython")
  if not os.path.exists(mjpython):
    sys.exit("the viewer needs mjpython on macOS; install mujoco in this environment and run: mjpython scripts/watchMatch.py")
  os.execv(mjpython, [mjpython, os.path.abspath(__file__), *sys.argv[1:]])

import carSoccer.config as cfg
from carSoccer.env import makeEnv
from carSoccer.learn.play import makeController, teamActions

def main():
  relaunchUnderMjpython()
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--policy", default="random", help="blue team: random, chase, idle or a checkpoint .pt")
  parser.add_argument("--opponent", default=None, help="orange team, same choices; default: same as --policy")
  parser.add_argument("--scatter", type=float, default=0.0, help="probability of a scattered kickoff")
  parser.add_argument("--speed", type=float, default=1.0, help="playback speed, 1 = real time")
  parser.add_argument("--randomize", action="store_true", help="draw a car variant and physics per match")
  parser.add_argument("--noise", action="store_true", help="realistic sensor noise (what the bots see)")
  parser.add_argument("--seed", type=int, default=0)
  args = parser.parse_args()

  game = replace(cfg.GameConfig(), actionRepeat=1)
  env = makeEnv(1, game=game, scatterProb=args.scatter, randomize=args.randomize, noise=args.noise,
                seed=args.seed, nthread=0)
  # one tick per frame, so every controller holds its decision for the action repeat it was built for
  hold = cfg.GameConfig().actionRepeat
  blue = makeController(args.policy, env, hold)
  orange = blue if args.opponent in (None, args.policy) else makeController(args.opponent, env, hold)

  obs = env.reset()
  model = env.engine.backend.models[0]       # world 0's own model, so the viewer shows its car variant
  data = mujoco.MjData(model)
  frame = env.engine.dt / args.speed
  teamReward = [0.0, 0.0]
  P = env.ctx.carsPerTeam

  def describe():
    p = env.state["params"]
    print(f"Match: car variant {p['variant'][0].item()}, floor friction {p['floorFriction'][0].item():.2f}, "
          f"ball {p['ballMass'][0].item():.3f} kg, latency {p['latency'][0].item()} ticks")

  describe()

  with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.lookat[:] = (0, 0, 0)
    viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 11.0, 90.0, -50.0
    while viewer.is_running():
      start = time.perf_counter()
      actions = blue(env.state, obs)
      actions = teamActions(actions, actions if orange is blue else orange(env.state, obs), env.ctx.carsPerTeam)
      obs, reward, events = env.step(actions)
      data.qpos[:] = env.engine.backend.qpos[0].numpy()
      data.qvel[:] = env.engine.backend.qvel[0].numpy()
      data.time = (env.state["clock"][0].item() + 1) * env.engine.dt   # never 0 here, so a viewer reset shows up as 0
      mujoco.mj_forward(model, data)
      viewer.sync()

      if data.time == 0:                     # the viewer's Reset button (or Backspace) cleared its data: restart the match
        obs = env.reset()
        teamReward = [0.0, 0.0]
        print("Reset: new match from kickoff")
        describe()
        continue

      teamReward[0] += reward[0, :P].sum().item()
      teamReward[1] += reward[0, P:].sum().item()
      if events["goal"][0].any():
        blueGoals, orangeGoals = env.state["score"][0].tolist()
        print(f"GOAL  blue {blueGoals} - {orangeGoals} orange   team reward so far: blue {teamReward[0]:+.1f}, orange {teamReward[1]:+.1f}")
      if events["done"][0]:
        blueGoals, orangeGoals = events["finalState", "score"][0].tolist()
        print(f"Full time  blue {blueGoals} - {orangeGoals} orange   team reward: blue {teamReward[0]:+.1f}, orange {teamReward[1]:+.1f}")
        teamReward = [0.0, 0.0]
        describe()
      time.sleep(max(0.0, frame - (time.perf_counter() - start)))

  env.close()

if __name__ == "__main__":
  main()
