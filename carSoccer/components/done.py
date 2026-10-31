from typing import Literal
from torch import Tensor
from carSoccer.engine.base import GameState, StepEvents

class DoneCondition:
  kind: Literal["terminated", "truncated"]

  def check(self, state: GameState, events: StepEvents) -> Tensor:
    """(N,) bool: the match is over."""
    raise NotImplementedError

class MatchClock(DoneCondition):
  """Matches end on the clock, so this is a truncation: the trainer bootstraps the value of the final state."""
  kind = "truncated"

  def __init__(self, seconds: float, controlHz: float):
    self.ticks = round(seconds * controlHz)

  def check(self, state: GameState, events: StepEvents) -> Tensor:
    return state["clock"] >= self.ticks
