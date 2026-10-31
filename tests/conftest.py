import mujoco
import pytest

import carSoccer.config as cfg
from carSoccer.engine.sim.scene import ball

@pytest.fixture
def make_model():
  """Factory: compile the ball-drop scene from (optional) configs."""
  def _make(ball_cfg=None, sim_cfg=None, drop_height=1.0):
    ball_cfg = ball_cfg or cfg.BallConfig()
    sim_cfg = sim_cfg or cfg.SimConfig()
    xml = ball.ball_drop_xml(ball_cfg, sim_cfg, drop_height)
    return mujoco.MjModel.from_xml_string(xml)
  
  return _make

@pytest.fixture
def model(make_model):
  """The default ball-drop model."""
  return make_model()