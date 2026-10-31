import csv, dataclasses, json, os, signal, time, torch, carSoccer.config as cfg
from dataclasses import dataclass, field, replace
from carSoccer.env import SoccerEnv, makeEnv
from carSoccer.learn.collector import Collector
from carSoccer.learn.ppo import PPO, PPOConfig
from carSoccer.learn.pool import OpponentPool
from carSoccer.models.policy import AgentModel, ModelSpec

try:
  from torch.utils.tensorboard import SummaryWriter
except ImportError:                        # tensorboard isn't in requirements.lock.txt; CSV logging still works
  SummaryWriter = None

@dataclass(frozen=True)
class TrainConfig:
  name: str = "selfplay"
  numWorlds: int = 256
  steps: int = 64                          # decisions per world per rollout
  iterations: int = 300
  seed: int = 0
  nthread: int | None = None               # MuJoCo rollout threads; default cores - 2
  torchThreads: int = 4
  matchSeconds: float = 10.0               # short matches: more kickoffs and variety per minute of play (v1 lesson)
  scatterProb: float = 1.0                 # random kickoffs, which v1 found far better than standard ones early on
  randomize: bool = True                   # car variant and physics per match
  noise: bool = True                       # realistic sensors for the actor; the critic sees the truth
  saveEvery: int = 10
  outDir: str = "runs"
  model: ModelSpec = ModelSpec()
  ppo: PPOConfig = PPOConfig()
  reward: cfg.RewardConfig = cfg.RewardConfig()
  poolCapacity: int = 8                    # snapshots remembered at once
  poolEvery: int = 10                      # iterations between adding the live policy to the pool
  poolProb: float = 0.3                    # chance a given iteration plays a pool snapshot at all
  poolFrac: float = 0.5                    # fraction of worlds handed to it, on iterations that do
  poolNoiseRange: tuple[float, float] = (0.85, 1.15)   # symmetric multiplier on its action std, see Collector
  tensorboard: bool = True

def configFromDict(d: dict) -> TrainConfig:
  d = dict(d)
  d["model"] = ModelSpec(**{k: tuple(v) if isinstance(v, list) else v for k, v in d["model"].items()})
  d["ppo"] = PPOConfig(**d["ppo"])
  d["reward"] = cfg.RewardConfig(**d["reward"])
  return TrainConfig(**d)

def buildEnv(train: TrainConfig, numWorlds: int | None = None, observe: bool = True, nthread: int | None = None, **overrides) -> SoccerEnv:
  game = replace(cfg.GameConfig(), matchSeconds=train.matchSeconds)
  kwargs = dict(game=game, scatterProb=train.scatterProb, randomize=train.randomize, noise=train.noise,
                seed=train.seed, nthread=train.nthread if nthread is None else nthread, observe=observe,
                rewardWeights=train.reward)
  kwargs.update(overrides)
  return makeEnv(numWorlds or train.numWorlds, **kwargs)

def buildModel(env: SoccerEnv, train: TrainConfig) -> AgentModel:
  return AgentModel(env.obsBuilder.spec, env.ctx.carsPerTeam, train.model)

def loadPolicy(path: str, env: SoccerEnv) -> AgentModel:
  """Model from a checkpoint, for watching or evaluation."""
  ckpt = torch.load(path, map_location="cpu", weights_only=False)
  model = buildModel(env, configFromDict(ckpt["config"]))
  model.load_state_dict(ckpt["model"])
  return model.eval()

class Learner:
  """Collect -> PPO update -> log -> checkpoint, one line of metrics per iteration (CSV in runs/<name>/progress.csv,
  plus live TensorBoard scalars if the tensorboard package is installed)."""

  def __init__(self, train: TrainConfig, resume: str | None = None):
    self.train = train
    torch.manual_seed(train.seed)
    torch.set_num_threads(train.torchThreads)
    self.env = buildEnv(train)
    self.model = buildModel(self.env, train)
    self.ppo = PPO(self.model, train.ppo)
    self.pool = OpponentPool(train.poolCapacity)
    self.collector = Collector(self.env, self.model, train.steps, train.ppo.gamma, train.ppo.lam,
                                pool=self.pool, poolProb=train.poolProb, poolFrac=train.poolFrac,
                                noiseRange=train.poolNoiseRange)
    self.iteration, self.envSteps = 0, 0
    self.dir = os.path.join(train.outDir, train.name)
    os.makedirs(self.dir, exist_ok=True)
    self.writer = SummaryWriter(self.dir) if train.tensorboard and SummaryWriter is not None else None
    if resume:
      self.load(resume)

  def learn(self) -> None:
    """Runs until train.iterations, or until stopped early: Ctrl-C, a SIGTERM (e.g. a cloud pre-emption notice), or
    another process touching <outDir>/<name>/STOP (scripts/stop.py - the "button" for a detached/remote run). Any
    of those finish the iteration in progress, save a checkpoint, and exit cleanly instead of losing it."""
    stopPath = os.path.join(self.dir, "STOP")
    stopRequested = {"flag": False}
    def requestStop(signum, frame):
      stopRequested["flag"] = True
    previousHandlers = {sig: signal.signal(sig, requestStop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
      csvPath = os.path.join(self.dir, "progress.csv")
      newFile = not os.path.exists(csvPath)
      with open(csvPath, "a", newline="") as f:
        writer = None
        while self.iteration < self.train.iterations:
          start = time.perf_counter()
          batch, stats = self.collector.collect()
          collectTime = time.perf_counter() - start
          metrics = self.ppo.update(batch)
          self.iteration += 1
          self.envSteps += self.train.numWorlds * self.train.steps * self.env.actionRepeat
          if self.iteration % self.train.poolEvery == 0:
            self.pool.add(self.model)                 # remember the current policy as a future opponent
          row = {"iteration": self.iteration, "envSteps": self.envSteps,
                 "envStepsPerSec": self.train.numWorlds * self.train.steps * self.env.actionRepeat / collectTime,
                 "updateSec": time.perf_counter() - start - collectTime, **stats, **metrics}
          if writer is None:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if newFile:
              writer.writeheader()
          writer.writerow(row)
          f.flush()
          self.report(row)
          if self.writer is not None:
            for key, value in row.items():
              if key != "iteration":
                self.writer.add_scalar(key, value, self.iteration)
            self.writer.flush()
          stopping = stopRequested["flag"] or os.path.exists(stopPath)
          if self.iteration % self.train.saveEvery == 0 or self.iteration == self.train.iterations or stopping:
            self.save()
          if stopping:
            print(f"stop requested - saved checkpoint at iteration {self.iteration}, exiting", flush=True)
            break
    finally:
      for sig, handler in previousHandlers.items():
        signal.signal(sig, handler)
      if os.path.exists(stopPath):
        os.remove(stopPath)                            # so a later --resume in this dir doesn't stop instantly
      self.env.close()
      if self.writer is not None:
        self.writer.close()

  def report(self, r: dict) -> None:
    print(f"it {r['iteration']:4d} | {r['envSteps'] / 1e6:6.2f}M steps {r['envStepsPerSec']:6.0f}/s upd {r['updateSec']:4.1f}s | "
          f"touch/car-min {r['touchesPerCarMin']:5.2f} goals/world-min {r['goalsPerWorldMin']:5.3f} "
          f"approach {r['approachSpeed']:+5.2f} m/s ball {r['ballSpeed']:4.2f} m/s | "
          f"kl {r['kl']:.4f} clip {r['clipFrac']:.2f} ent {r['entropy']:+.2f} std {r['std']:.2f} ev {r['explainedVariance']:+.2f}",
          flush=True)

  def save(self) -> None:
    ckpt = {
      "model": self.model.state_dict(), "optimizer": self.ppo.optimizer.state_dict(),
      "iteration": self.iteration, "envSteps": self.envSteps, "config": dataclasses.asdict(self.train),
      "returnStd": vars(self.collector.returnStd), "pool": self.pool.snapshots,
    }
    path = os.path.join(self.dir, f"ckpt{self.iteration:05d}.pt")
    torch.save(ckpt, path)
    torch.save(ckpt, os.path.join(self.dir, "latest.pt"))
    with open(os.path.join(self.dir, "config.json"), "w") as f:
      json.dump(dataclasses.asdict(self.train), f, indent=2)

  def load(self, path: str) -> None:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    self.model.load_state_dict(ckpt["model"])
    self.ppo.optimizer.load_state_dict(ckpt["optimizer"])
    self.iteration, self.envSteps = ckpt["iteration"], ckpt["envSteps"]
    vars(self.collector.returnStd).update(ckpt["returnStd"])
    self.pool.snapshots = ckpt.get("pool", [])            # older checkpoints had no pool: resume with it empty
