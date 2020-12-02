import jax.numpy as np
from jax import jit, mask

def bucket_jit(f):
  compiled_f = jit(mask(f, ['n'], ''))
  def wrapped(x):
    amount = 128 - x.shape[0] % 128
    padded_x = np.pad(x, (0, amount))
    return compiled_f([padded_x], dict(n=x.shape[0]))
  return wrapped

@bucket_jit
def foo(x):
  print("recompiling!")  # actually retracing, but effectively correct
  return np.tanh(np.sum(x))

foo(np.arange(4))  # recompiling!
foo(np.arange(5))
foo(np.arange(6))
foo(np.arange(300))  # recompiling!

from jax import grad
grad(foo)(np.arange(3.))  # recompiling!
grad(foo)(np.arange(4.))
grad(foo)(np.arange(129.))  # recompiling!
