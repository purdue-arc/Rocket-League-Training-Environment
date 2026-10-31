import copy
import torch
from carSoccer.models.policy import AgentModel

class OpponentPool:
  """Ring buffer of frozen past-policy snapshots. Pure mirror self-play plays the live policy against itself every
  match, which has no memory: if it drifts away from a weakness an earlier version of itself exploited, nothing
  stops it drifting back (strategy cycling). Sampling an old snapshot as one side's opponent some of the time keeps
  old lessons live, at the cost of a slice of training data that isn't on-policy (see Collector's "trainable" mask)."""

  def __init__(self, capacity: int = 8):
    self.capacity = capacity
    self.snapshots: list[dict] = []

  def add(self, model: AgentModel) -> None:
    state = copy.deepcopy(model.state_dict())
    if len(self.snapshots) < self.capacity:
      self.snapshots.append(state)
    else:
      self.snapshots[torch.randint(0, self.capacity, (1,)).item()] = state   # random eviction: keeps old and new mixed

  def sample(self, template: AgentModel) -> AgentModel:
    """A frozen copy of template's architecture loaded with a randomly chosen past snapshot's weights."""
    idx = torch.randint(0, len(self.snapshots), (1,)).item()
    opponent = copy.deepcopy(template)
    opponent.load_state_dict(self.snapshots[idx])
    opponent.eval()
    for p in opponent.parameters():
      p.requires_grad_(False)
    return opponent

  def __len__(self) -> int:
    return len(self.snapshots)
