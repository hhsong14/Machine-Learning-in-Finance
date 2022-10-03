import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import optax
import numpy as np

train_x = np.load("X_train.npy")
val_x = np.load("X_val.npy")


y_train = np.load("y_train.npy")
y_val = np.load("y_val.npy")

X_train = np.column_stack([np.ones_like(y_train),train_x])
X_val = np.column_stack([np.ones_like(y_val),val_x])

theta = np.zeros(X_train.shape[1])

def mse(a, b):
    return ((a - b)**2).mean()


optimizer = optax.adam
n_models = 10

opt_state = optimizer(1.).init(0)
theta = jnp.stack([theta] * n_models)
opt_state = jax.tree_map(lambda item: jnp.stack([item] * n_models), opt_state)


batch_size = 500

def sample():
  i = np.random.choice(8000, batch_size)
  return X_train[i], y_train[i]


lr = jnp.array([0.1, 1e-2, 1e-3, 1e-4])
regularization = jnp.array([1e3, 1e2, 1e1, 1e-1, 1e-2, 1e-4, 1e-6])

lr_try = np.random.choice(lr, n_models)
regularization_try = np.random.choice(regularization, n_models)


@jax.jit
@jax.vmap
def update(theta, opt_state, lr, regularization):
    def L(theta):
        X_train ,y_train = sample()
        predict = X_train @ theta
        return mse(predict, y_train) + regularization * theta @ theta

    grads = jax.grad(L)(theta)
    updates, opt_state = optimizer(lr).update(grads, opt_state)
    theta = optax.apply_updates(theta, updates)
    return theta, opt_state


@jax.jit
@jax.vmap
def evaluate(theta):
    return mse(X_val @ theta, y_val)


for iteration in range(10000):
    print(iteration)
    theta, opt_state = update(theta, opt_state, lr_try, regularization_try)



idx_best = jnp.argmin(evaluate(theta))

def predict(X):
    return X @  theta[idx_best]




print(mse(y_val,predict(X_val)))
