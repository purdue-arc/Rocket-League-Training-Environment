import csv, dataclasses, json, os, time, torch, carSoccer.config as cfg
from dataclasses import dataclass, field, replace
from carSoccer.env import SoccerEnv, makeEnv
from carSoccer.learn.collector import Collector
from carSoccer.learn.ppo import PPO, PPOConfig
from carSoccer.models.policy import AgentModel, ModelSpec

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
  """Collect -> PPO update -> log -> checkpoint, one line of metrics per iteration (CSV in runs/<name>/progress.csv)."""

  def __init__(self, train: TrainConfig, resume: str | None = None):
    self.train = train
    torch.manual_seed(train.seed)
    torch.set_num_threads(train.torchThreads)
    self.env = buildEnv(train)
    self.model = buildModel(self.env, train)
    self.ppo = PPO(self.model, train.ppo)
    self.collector = Collector(self.env, self.model, train.steps, train.ppo.gamma, train.ppo.lam)
    self.iteration, self.envSteps = 0, 0
    self.dir = os.path.join(train.outDir, train.name)
    os.makedirs(self.dir, exist_ok=True)
    if resume:
      self.load(resume)

  def learn(self) -> None:
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
        if self.iteration % self.train.saveEvery == 0 or self.iteration == self.train.iterations:
          self.save()
    self.env.close()

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
      "returnStd": vars(self.collector.returnStd),
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
