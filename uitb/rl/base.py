from abc import ABC, abstractmethod

class BaseRLModel(ABC):

  def __init__(self, **kwargs):
    pass

  @abstractmethod
  def learn(self, wandb_callback):
    pass