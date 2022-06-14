from .base import BaseEffortModel


class Zero(BaseEffortModel):

  def cost(self, model, data):
    return 0

  def reset(self, model, data):
    pass

  def update(self, model, data):
    pass