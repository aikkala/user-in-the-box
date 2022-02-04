from abc import ABC, abstractmethod

class BaseModel(ABC):

  def __init__(self, config):
    self.config = config

  @abstractmethod
  def learn(self, wandb_callback):
    pass