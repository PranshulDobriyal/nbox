from typing import Sequence
import numpy as np
import jax
import flax.linen as nn
from nbox.model import Model
import time

def test_feedforward():
  class MLP(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
      for feat in self.features[:-1]:
        x = nn.relu(nn.Dense(feat)(x))
      x = nn.Dense(self.features[-1])(x)
      x = nn.Dense(features=2)(x)
      x = nn.log_softmax(x)
      return x

  model = MLP([12, 8, 4])
  batch = np.random.rand(2, 10)
  variables = model.init(jax.random.PRNGKey(0), batch)

  m = Model(model, variables)
  return m(batch).outputs

def run_fn(name, fn):
    def br():
        print("#"*70, "\n")
    start_time = time.time()
    out = fn()
    end_time = time.time()
    br()
    print(f"{name}: \n {out}\n\nThe Function took {end_time-start_time} seconds to run")
    br()

#Test Feedforward - 
run_fn("FeedForward", test_feedforward)