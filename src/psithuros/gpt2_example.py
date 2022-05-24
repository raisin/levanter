import math
from functools import partial
from typing import Callable, TypeVar

import chex
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrandom
import optax

import equinox as eqx
from optax import OptState
from transformers import GPT2Config

from palm_lite import PaLM
from psithuros.gpt2 import Gpt2LMHeadModel

NUM_TOKENS = 2048
SEQ_LEN = 512

def dataloader(arrays, batch_size, *, key):
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = jnp.arange(dataset_size)
    while True:
        perm = jrandom.permutation(key, indices)
        (key,) = jrandom.split(key, 1)
        start = 0
        end = batch_size
        while end < dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size


def get_data(dataset_size, *, key):
    k_x, k_y = jrandom.split(key, 2)
    x = jrandom.randint(k_x, [dataset_size, SEQ_LEN], minval=0, maxval=NUM_TOKENS)
    # y = jrandom.randint(k_y, [dataset_size, SEQ_LEN], minval=0, maxval=NUM_TOKENS)
    y = jnp.concatenate( [x[:, 1:], jnp.zeros((dataset_size, 1), dtype=jnp.int32)], axis=1)

    return x, y


def main(
    dataset_size=10000,
    batch_size=32,
    learning_rate=3e-3,
    steps=200,
    seed=5678,
):
    data_key, loader_key, model_key, training_key = jrandom.split(jrandom.PRNGKey(seed), 4)
    xs, ys = get_data(dataset_size, key=data_key)
    iter_data = dataloader((xs, ys), batch_size, key=loader_key)

    config = GPT2Config(vocab_size=NUM_TOKENS, n_positions=SEQ_LEN, n_embd=128, n_ctx=SEQ_LEN, n_layer=4, n_head=4, n_embd_shared_axes=0, hidden_dim=128, num_attention_heads=4, intermediate_size=1024, hidden_act="gelu", hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=SEQ_LEN, type_vocab_size=2, initializer_range=0.02)

    model = Gpt2LMHeadModel(config, key=model_key)

    @eqx.filter_value_and_grad
    def compute_loss(model, x, y, *, inference, key):
        model = partial(model, inference=inference, key=key)
        pred_y = jax.vmap(model)(x)
        return jnp.mean(optax.softmax_cross_entropy(pred_y, jax.nn.one_hot(y, num_classes=NUM_TOKENS)))
        # return jnp.mean(optax.softmax_cross_entropy(pred_y, pred_y))

    # Important for efficiency whenever you use JAX: wrap everything into a single JIT
    # region
    @eqx.filter_jit
    def make_step(model, x, y, opt_state: OptState, *, inference, key):
        loss, grads = compute_loss(model, x, y, inference=inference, key=key)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    optim = optax.adam(learning_rate)
    opt_state = optim.init(model)
    keys = jax.random.split(training_key, steps)
    for step, (x, y), k in zip(range(steps), iter_data, keys):
        loss, model, opt_state = make_step(model, x, y, opt_state, inference=True, key=k)
        loss = loss.item()
        print(f"step={step}, loss={loss}")

    pred_ys = jax.vmap(model)(xs)
    num_correct = jnp.sum((pred_ys > 0.5) == ys)
    final_accuracy = (num_correct / dataset_size).item()
    print(f"final_accuracy={final_accuracy}")


if __name__ == "__main__":
    main()