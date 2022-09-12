import copy
import logging as pylogging
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
from optax import MultiStepsState
from tqdm import tqdm

import wandb
from levanter.jax_utils import jnp_to_python
from levanter.trainer_hooks import StepInfo


def log_optimizer_hyperparams(opt_state, prefix: Optional[str] = None, *, step=None):
    if isinstance(opt_state, MultiStepsState):
        opt_state = opt_state.inner_opt_state

    def wrap_key(key):
        if prefix:
            return f"{prefix}/{key}"
        return key

    if hasattr(opt_state, "hyperparams"):
        # we insert the mean because when we replicate the optimization state, the optimizer state is copied along with
        # any hyperparams...
        params = {wrap_key(k): jnp_to_python(jnp.mean(v)) for k, v in opt_state.hyperparams.items()}
        # print(params)
        wandb.log(params, step=step)


def log_performance_stats(
    tokens_per_example: int,
    batch_size: int,
    flops_per_example: Optional[float] = None,
    prefix: Optional[str] = "throughput",
):
    def wrap_key(key):
        if prefix:
            return f"{prefix}/{key}"
        return key

    def log_performance_stats(step_info: StepInfo):
        if step_info.step_duration != 0.0:
            wandb.log(
                {
                    wrap_key("examples_per_second"): float(batch_size) / step_info.step_duration,
                    wrap_key("tokens_per_second"): float(tokens_per_example) / step_info.step_duration * batch_size,
                    wrap_key("duration"): step_info.step_duration,
                },
                step=step_info.step,
            )

            if flops_per_example is not None:
                wandb.log(
                    {
                        wrap_key("gflops_per_second"): flops_per_example / 1e9 / step_info.step_duration * batch_size,
                    },
                    step=step_info.step,
                )

    return log_performance_stats


def pbar_logger(iterable=None, desc="train", **tqdm_mkwargs):
    kwargs = copy.copy(tqdm_mkwargs)
    if "desc" not in kwargs:
        kwargs["desc"] = desc
    if "iterable" not in kwargs:
        kwargs["iterable"] = iterable
    pbar = tqdm(**kwargs)

    def update_pbar(step: StepInfo):
        pbar.update(step.step - pbar.n)
        pbar.set_postfix(loss=step.loss)

    return update_pbar


def log_to_wandb(step: StepInfo):
    wandb.log({"train/loss": step.loss}, step=step.step)
    log_optimizer_hyperparams(step.opt_state, step=step.step)


def init_logger(path: Path, level: int = pylogging.INFO) -> None:
    """
    Initialize logging.Logger with the appropriate name, console, and file handlers.

    :param path: Path for writing log file
    :param level: Default logging level
    """
    process_index = jax.process_index()
    log_format = f"%(asctime)s - {process_index} - %(name)s - %(filename)s:%(lineno)d - %(levelname)s :: %(message)s"
    file_handler = pylogging.FileHandler(path, mode="a")
    file_handler.setFormatter(pylogging.Formatter(log_format))

    # Create Root Logger w/ Base Formatting
    pylogging.basicConfig(level=level, format=log_format, handlers=[file_handler])
