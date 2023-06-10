from jax.random import PRNGKey

from haliax import Axis
from levanter.models.backpack import BackpackConfig, BackpackLMHeadModel
from levanter.models.gpt2 import Gpt2Config, Gpt2LMHeadModel
from levanter.utils.jax_utils import parameter_count

GPT2_NAME = "gpt2"
BACKPACK_NAME = "backpack"


MODEL = {
    GPT2_NAME: Gpt2LMHeadModel,
    BACKPACK_NAME: BackpackLMHeadModel,
}
VOCAB_SIZE = {
    GPT2_NAME: 50257,
    BACKPACK_NAME: 50264,
}


def get_param_count(
    model_name=GPT2_NAME,
    seq_len=1024,
    hidden_dim=768,
    num_layers=12,
    num_heads=12,
    num_senses=16,
    sense_intermediate_scale=4,
):
    assert model_name in MODEL.keys(), f"model_name must be one of {MODEL.keys()}"
    if model_name == GPT2_NAME:
        config = Gpt2Config(
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            gradient_checkpointing=False,
        )
    elif model_name == BACKPACK_NAME:
        config = BackpackConfig(
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            num_senses=num_senses,
            sense_intermediate_scale=sense_intermediate_scale,
        )
    key = PRNGKey(0)
    Vocab = Axis("vocab", VOCAB_SIZE[model_name])
    model = MODEL[model_name].init(Vocab, config, key=key)
    print(f"parameter count: {parameter_count(model):,}")


if __name__ == "__main__":
    get_param_count(
        model_name=BACKPACK_NAME,
        num_senses=48,
        num_layers=36,
        num_heads=20,
        sense_intermediate_scale=7,
        hidden_dim=1280,
        seq_len=512,
    )
