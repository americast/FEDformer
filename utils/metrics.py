import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def MASE(actual, predicted):
  """
  Calculates the mean absolute scaled error (MASE) for a given set of actual and predicted values.

  Args:
    actual: The actual values.
    predicted: The predicted values.

  Returns:
    The MASE value.
  """

  naive_forecast = np.copy(actual[:-1])
  naive_mae = np.mean(np.abs(actual[1:] - naive_forecast))
  mase_value = np.mean(np.abs(actual - predicted)) / naive_mae
  return mase_value

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    mase = MASE(true, pred)

    return mae, mse, rmse, mape, mspe, mase
