
from llm_project.configs.base import ModelConfig
from llm_project.experiments.multi import build_multi_head_model
from llm_project.experiments.single import build_single_head_model


def build_model(cfg: ModelConfig):
    return build_multi_head_model(cfg)