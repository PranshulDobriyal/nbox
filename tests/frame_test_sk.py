import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import nbox
import numpy as np
from nbox.framework.on_ml import SklearnInput
from nbox.model import Model
import time


def br():
    print("\n")
    print("#"*50, "\n")


def test_feedforward():
  from sklearn.neural_network import MLPClassifier
  X = np.random.uniform(size=(100, 224))
  Y = np.random.uniform(size=(100, 1))
  Y[Y>0.5] = 1
  Y[Y<=0.5] = 0
  mlp = MLPClassifier(hidden_layer_sizes=(2))
  mlp.fit(X,Y)

  x = np.random.uniform(size = (10, 224))
  inputs = SklearnInput(
      inputs = x, 
      method = "predict", 
      kwargs = None,
  )
  model = Model(mlp, None)
  first_out = model(inputs).outputs

  new_model = nbox.Model.deserialise(
    model.serialise(
      input_object = inputs,
      model_name = "test69",
      export_type = "pkl",
      _unit_test = True
    )
  )
  second_out = new_model(inputs).outputs
  assert np.array_equal(first_out, second_out)
  return second_out


def test_random_forest():
  def get_model():
      iris = load_iris()
      X, y = iris.data, iris.target
      X_train, X_test, y_train, y_test = train_test_split(X, y)
      clr = RandomForestClassifier()
      clr.fit(X_train, y_train)
      return clr, X_test, y_test

  _m, _x, _y = get_model()
  model = nbox.Model(_m, None)

  inputs = SklearnInput(
      inputs = _x, 
      method = "predict_proba", 
      kwargs = None,

  )
  first_out = model(inputs).outputs

  new_model = nbox.Model.deserialise(
    model.serialise(
      input_object = inputs,
      model_name = "test69",
      export_type = "pkl",
      _unit_test = True
    )
  )
  second_out = new_model(inputs).outputs
  assert np.array_equal(first_out, second_out)
  return second_out[:5]


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

#Test Random Forest -
run_fn("Random Forest", test_random_forest)