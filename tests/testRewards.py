import pytest, torch

import carSoccer.config as cfg
from carSoccer.components.rewards import GameplayReward
from carSoccer.components.roster import Roster
from carSoccer.engine.base import emptyEvents
from carSoccer.env import makeEnv

P = cfg.SceneConfig().carsPerTeam
W = cfg.RewardConfig()

@pytest.fixture(scope="module")
def env():
  env = makeEnv(2, nthread=0, observe=False)
  yield env
  env.close()

def events(N=2, touch=None, goal=None):
  ev = emptyEvents(N, P, torch.device("cpu"))
  if touch is not None:
    ev["ballTouch"] = torch.tensor(touch)
  if goal is not None:
    ev["goal"] = torch.tensor(goal)
  return ev

# ---------- reward terms ----------

def testTouchPaysTheCarThatTouched(env):
  reward, terms = GameplayReward(1, env.ctx)(events(1, touch=[[False, True, False, False, False, False]]))
  assert reward[0].tolist() == [0, W.touch, 0, 0, 0, 0]

def testOnlyNewTouchesPay(env):
  fn = GameplayReward(1, env.ctx)
  holding = events(1, touch=[[True, False, False, False, False, False]])
  assert fn(holding)[0][0, 0] == W.touch
  assert fn(holding)[0][0, 0] == 0                                     # still pinned against the ball: nothing
  fn(events(1))
  assert fn(holding)[0][0, 0] == W.touch                               # let go and touched again

def testKickoffForgetsTouches(env):
  fn = GameplayReward(1, env.ctx)
  holding = events(1, touch=[[True] + [False] * 5])
  fn(holding)
  fn.reset(torch.tensor([True]))
  assert fn(holding)[0][0, 0] == W.touch

def testGoalPaysWholeTeamAndCostsTheOther(env):
  reward, terms = GameplayReward(2, env.ctx)(events(goal=[[True, False], [False, True]]))
  assert reward[0].tolist() == [W.goal] * P + [-W.conceded] * P      # blue scored in world 0
  assert reward[1].tolist() == [-W.conceded] * P + [W.goal] * P      # orange scored in world 1
  assert terms["goal"].sum() == 2 * P and terms["conceded"].sum() == 2 * P

def testGoalOutweighsTouches():
  assert W.goal >= 5 * W.touch

def testEnvRewardsAGoal(env):
  env.reset()
  X, r = cfg.ArenaConfig().length / 2, cfg.BallConfig().radius
  target = env.state.clone()
  target["ball", "pos"][:] = torch.tensor([X + r + 0.05, 0, r])        # in orange's goal: blue scores
  env.engine.setState(torch.ones(2, dtype=torch.bool), target)
  _, reward, ev = env.step(torch.zeros(2, 2 * P, 2))
  assert ev["goal"][:, 0].all()
  assert (reward[:, :P] >= W.goal).all() and (reward[:, P:] <= -W.conceded).all()

def testEnvRewardsTouchesInPlay(env):
  env.reset()
  push = torch.zeros(2, 2 * P, 2)
  push[:, 0, 0] = 1.0                                                  # blue0 drives straight at the ball from kickoff
  touches = torch.zeros(2, 2 * P)
  for _ in range(40):
    _, _, ev = env.step(push)
    touches += ev["rewardTerms", "touch"]
  assert (touches[:, 0] >= 1).all() and touches[:, 1:P].sum() == 0     # idle teammates earn nothing

# ---------- roster ----------

def testRosterSlicesAndCombine():
  roster = Roster(8, P, [(0.75, "learner", "learner"), (0.25, "learner", "scripted")], torch.device("cpu"))
  assert roster.cars("learner").sum() == 6 * 6 + 2 * 3
  assert roster.cars("scripted")[6:, P:].all() and not roster.cars("scripted")[:6].any()
  actions = roster.combine({"learner": torch.ones(8, 2 * P, 2), "scripted": -torch.ones(8, 2 * P, 2)})
  assert actions[7, :P].eq(1).all() and actions[7, P:].eq(-1).all() and actions[0].eq(1).all()
