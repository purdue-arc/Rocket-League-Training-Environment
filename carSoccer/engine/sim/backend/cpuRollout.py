import os, mujoco, numpy as np, torch
from mujoco import rollout
from torch import Tensor

STATE_SPEC = mujoco.mjtState.mjSTATE_FULLPHYSICS
CONTROL_SPEC = mujoco.mjtState.mjSTATE_CTRL | mujoco.mjtState.mjSTATE_XFRC_APPLIED

def stateSlices(model: mujoco.MjModel, spec: int) -> dict[int, slice]:
  """Column range of each component inside a state vector of the given spec; mj_getState writes components in bit order."""
  slices, offset = {}, 0
  for i in range(mujoco.mjtState.mjNSTATE.value):
    bit = 1 << i
    if spec & bit:
      size = mujoco.mj_stateSize(model, bit)
      slices[bit] = slice(offset, offset + size)
      offset += size

  return slices

class CpuRolloutBackend:
  """N worlds stepped by mujoco.rollout's C++ thread pool: one Python call per env step for all worlds."""

  def __init__(self, model: mujoco.MjModel, numWorlds: int, nthread: int | None = None):
    self.model, self.numWorlds, self.device = model, numWorlds, torch.device("cpu")
    nthread = max(1, (os.cpu_count() or 4) - 2) if nthread is None else nthread
    self.nthread = min(nthread, numWorlds)
    self.pool = rollout.Rollout(nthread=self.nthread)
    self.datas = [mujoco.MjData(model) for _ in range(max(1, self.nthread))]
    self.models = [model] * numWorlds        # replaced by a private copy per world the first time setParams touches it

    # state: one (N, nstate) array that is both the rollout's initial state and, after each call, its result
    data = mujoco.MjData(model)
    initial = np.empty(mujoco.mj_stateSize(model, STATE_SPEC))
    mujoco.mj_getState(model, data, initial, STATE_SPEC)
    self.stateNp = np.tile(initial, (numWorlds, 1))
    cols = stateSlices(model, STATE_SPEC)
    qposCols, qvelCols = cols[mujoco.mjtState.mjSTATE_QPOS], cols[mujoco.mjtState.mjSTATE_QVEL]
    self.qpos = torch.from_numpy(self.stateNp[:, qposCols])
    self.qvel = torch.from_numpy(self.stateNp[:, qvelCols])
    self.qpos0 = torch.from_numpy(initial[qposCols].copy())

    # controls: (N, 1, ncontrol) tiled across substeps by rollout; ctrl and xfrc are views into it
    self.control = np.zeros((numWorlds, 1, mujoco.mj_stateSize(model, CONTROL_SPEC)))
    self.ctrl = torch.from_numpy(self.control[:, 0, :model.nu])
    self.xfrc = torch.from_numpy(self.control[:, 0, model.nu:].reshape(numWorlds, model.nbody, 6))

    self.sensorPeak = np.zeros((numWorlds, model.nsensordata))
    self.sensordata = torch.from_numpy(self.sensorPeak)
    self.outState, self.outSensor = None, None

  def step(self, ctrl: Tensor, xfrc: Tensor | None, nsub: int) -> None:
    self.ctrl.copy_(ctrl)
    if xfrc is None:
      self.xfrc.zero_()
    else:
      self.xfrc.copy_(xfrc)
    if self.outState is None or self.outState.shape[1] != nsub:
      self.outState = np.empty((self.numWorlds, nsub, self.stateNp.shape[1]))
      self.outSensor = np.empty((self.numWorlds, nsub, self.model.nsensordata))

    self.pool.rollout(self.models, self.datas, self.stateNp, self.control, control_spec=CONTROL_SPEC,
                      nstep=nsub, state=self.outState, sensordata=self.outSensor)
    self.stateNp[:] = self.outState[:, -1]   # in place, so the qpos / qvel views stay valid
    np.max(self.outSensor, axis=1, out=self.sensorPeak)

  def writeState(self, mask: Tensor, qpos: Tensor, qvel: Tensor) -> None:
    m = mask[:, None]
    self.qpos.copy_(torch.where(m, qpos, self.qpos))
    self.qvel.copy_(torch.where(m, qvel, self.qvel))

  def setParams(self, mask: Tensor, params: dict[str, Tensor]) -> None:
    # runs at resets only; on CPU every world owns its model, so writing a world's parameters is a per-world copy
    for w in torch.nonzero(mask).flatten().tolist():
      if self.models[w] is self.model:
        self.models[w] = self.model.__copy__()
      for name, value in params.items():
        getattr(self.models[w], name)[...] = value[w].numpy()

  def close(self) -> None:
    self.pool.close()
