from dataclasses import replace

import torch, carSoccer.config as cfg
from tensordict import TensorDict
from torch import Tensor
from carSoccer.components.actions import ActionParser, ContinuousCarActions
from carSoccer.components.ctx import Ctx
from carSoccer.components.done import DoneCondition, MatchClock
from carSoccer.components.mutators import Kickoff, NewMatch, RandomPhysics, SampleCarVariant, StateMutator
from carSoccer.components.obs.builders import EntityCarObs, ObsBuilder
from carSoccer.components.rewards import GameplayReward
from carSoccer.engine.base import StepEvents, emptyEvents, maskedAssign
from carSoccer.engine.sim.scene.builder import buildScene
from carSoccer.engine.sim.simEngine import SimEngine

class SoccerEnv:
  """Engine + components + action repeat. One step() is one decision for every world; finished worlds are reset
  before it returns (auto-reset) and their last state is kept in events["finalState"]."""

  def __init__(self, engine, actionParser: ActionParser, obsBuilder: ObsBuilder | None,
               mutators: list[StateMutator], doneConditions: list[DoneCondition], ctx: Ctx, actionRepeat: int,
               reward: GameplayReward | None = None):
    self.engine, self.actionParser, self.obsBuilder, self.reward = engine, actionParser, obsBuilder, reward
    self.mutators, self.doneConditions, self.ctx, self.actionRepeat = mutators, doneConditions, ctx, actionRepeat
    self.numWorlds, self.device = engine.numWorlds, engine.device

  @property
  def state(self):
    return self.engine.state

  def reset(self) -> TensorDict | None:
    everyWorld = torch.ones(self.numWorlds, dtype=torch.bool, device=self.device)
    self.startWorlds(everyWorld, everyWorld)
    return self.observe(self.engine.state)

  def startWorlds(self, newMatch: Tensor, kickoff: Tensor) -> None:
    """Run the mutators on worlds that start a match or need a kickoff, then have the engine apply the result."""
    target = self.engine.state.clone()
    for m in self.mutators:
      m.apply(target, newMatch if m.on == "match" else newMatch | kickoff, self.ctx)
    if self.obsBuilder is not None:
      self.obsBuilder.reset(newMatch | kickoff)
    if self.reward is not None:
      self.reward.reset(newMatch | kickoff)
    if any(m.setsParams for m in self.mutators):
      self.engine.setParams(newMatch, target["params"])         # new car variant and physics, before placing the cars
    self.engine.setState(newMatch | kickoff, target)

  def step(self, rawActions: Tensor) -> tuple[TensorDict | None, Tensor, StepEvents]:
    """One decision. Returns (observations (N * 2P agents), reward (N, 2P), events (N,))."""
    carActions = self.actionParser.parse(rawActions, self.ctx)
    N, P = self.numWorlds, self.ctx.carsPerTeam
    events = emptyEvents(N, P, self.device)
    frozen = torch.zeros(N, dtype=torch.bool, device=self.device)
    finalState = self.engine.state.clone()

    for _ in range(self.actionRepeat):
      tick = self.engine.step(torch.where(frozen[:, None, None], torch.zeros_like(carActions), carActions))
      live = ~frozen
      for key in ("goal", "ballTouch", "ballOut", "flipped"):
        e = tick[key]
        events[key] |= e & live.view(-1, *[1] * (e.dim() - 1))
      terminated = torch.zeros_like(frozen)
      truncated = torch.zeros_like(frozen)
      for d in self.doneConditions:
        hit = d.check(self.engine.state, tick)
        if d.kind == "truncated":
          truncated |= hit
        else:
          terminated |= hit
      newlyDone = (terminated | truncated) & live
      events["done"] |= newlyDone
      events["truncated"] |= truncated & ~terminated & live
      maskedAssign(finalState, self.engine.state, newlyDone)
      frozen |= newlyDone                    # nothing happens after the final whistle

    kickoff = (events["goal"].any(-1) | events["ballOut"]) & ~events["done"]
    events["kickoff"] = kickoff | events["done"]
    events["finalState"] = finalState
    if self.obsBuilder is not None:
      events["finalObs"] = self.obsBuilder.build(finalState, self.ctx).reshape(N, -1)   # (N, agents per world)
    reward = torch.zeros(N, 2 * P, device=self.device)
    if self.reward is not None:
      reward, events["rewardTerms"] = self.reward(events)
    self.startWorlds(events["done"], kickoff)
    return self.observe(self.engine.state), reward, events

  def observe(self, state) -> TensorDict | None:
    return None if self.obsBuilder is None else self.obsBuilder.build(state, self.ctx)

  def close(self) -> None:
    self.engine.close()

def makeEnv(numWorlds: int, sceneCfg: cfg.SceneConfig = cfg.SceneConfig(), game: cfg.GameConfig = cfg.GameConfig(),
            scatterProb: float = 0.0, randomize: bool = False, noise: bool = False, seed: int = 0,
            nthread: int | None = None, observe: bool = True, rewardWeights: cfg.RewardConfig = cfg.RewardConfig()) -> SoccerEnv:
  """The default 3v3 env: car scheme, continuous actions, standard (or scattered) kickoffs, matches end on the clock,
  gameplay reward. randomize draws a car variant and physics per match; noise uses realistic sensors."""
  if noise:
    game = replace(game, sensors=cfg.SensorConfig.realistic())
  generator = torch.Generator().manual_seed(seed)
  engine = SimEngine(buildScene(sceneCfg), sceneCfg, game, numWorlds, nthread, generator=generator)
  ctx = Ctx(
    generator=generator, device=engine.device, dt=engine.dt,
    arena=sceneCfg.arena, ballRadius=sceneCfg.ball.radius, carsPerTeam=sceneCfg.carsPerTeam,
    spawnHeights=engine.spawnHeights, matchTicks=round(game.matchSeconds * game.controlHz),
  )
  randomizers = [SampleCarVariant(), RandomPhysics(sceneCfg)] if randomize else []
  return SoccerEnv(
    engine, ContinuousCarActions(), EntityCarObs(ctx) if observe else None,
    mutators=[NewMatch(), *randomizers, Kickoff(scatterProb)],              # variant before Kickoff: spawn height
    doneConditions=[MatchClock(game.matchSeconds, game.controlHz)],
    ctx=ctx, actionRepeat=game.actionRepeat, reward=GameplayReward(numWorlds, ctx, rewardWeights),
  )
