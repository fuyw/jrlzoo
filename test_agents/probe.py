from flax import linen as nn
from flax.training import train_state
import jax
import jax.numpy as jnp
import optax
import numpy as np
from tqdm import trange
from sklearn.model_selection import train_test_split, KFold


class MLP(nn.Module):
    output_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.output_dim, name="probe_fc")(x)
        return x


class ProbeTrainer:
    def __init__(self, input_dim, output_dim, batch_size=128, lr=3e-4):
        self.mlp = MLP(output_dim)
        rng = jax.random.PRNGKey(0)
        dummy_inputs = jnp.ones(input_dim)
        params = self.mlp.init(rng, dummy_inputs)["params"]
        self.state = train_state.TrainState.create(
            apply_fn=self.mlp.apply,
            params=params,
            tx=optax.adam(learning_rate=lr))
        self.batch_size = batch_size

    def train(self, X, Y):
        @jax.jit
        def loss_fn(params, x, y):
            output = self.mlp.apply({"params": params}, x)
            loss = (output - y)**2
            return loss.mean()
        grad_fn = jax.value_and_grad(loss_fn)

        kf = KFold(n_splits=5)
        kf_losses = []
        for trainval_idx, test_idx in kf.split(X):
            trainval_X, test_X = X[trainval_idx], X[test_idx]
            train_idx, valid_idx = train_test_split(trainval_idx, test_size=0.1)
            train_X, valid_X, test_X = X[train_idx], X[valid_idx], X[test_idx]
            train_Y, valid_Y, test_Y = Y[train_idx], Y[valid_idx], Y[test_idx]
            
            batch_num = int(np.ceil(len(train_idx) / self.batch_size))
            min_valid_loss = np.inf
            optimal_params = None
            patience = 0
            for epoch in trange(100, desc=f"CV #{len(kf_losses)+1}"):
                epoch_loss = 0
                for i in range(batch_num):
                    batch_idx = train_idx[i*self.batch_size:(i+1)*self.batch_size]
                    batch_x = X[batch_idx]
                    batch_y = Y[batch_idx]
                    loss, grad = grad_fn(self.state.params, batch_x, batch_y)
                    self.state = self.state.apply_gradients(grads=grad)
                    epoch_loss += loss.item()
                valid_loss = loss_fn(self.state.params, valid_X, valid_Y).item()
                if valid_loss < min_valid_loss:
                    min_valid_loss = valid_loss
                    patience = 0
                    optimal_params = self.state.params
                else:
                    patience += 1
                print(f'# Epoch {epoch}: train_loss: {epoch_loss/batch_num:6f}, valid_loss: {valid_loss:.6f}')
                if patience == 10:
                    print(f'Early break at epoch {epoch}.')
                    break

            # test
            test_loss = loss_fn(optimal_params, test_X, test_Y).item()
            kf_losses.append(test_loss)
        return kf_losses
