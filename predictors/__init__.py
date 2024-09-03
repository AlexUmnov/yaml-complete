from functools import partial

from predictors.proprietary import OpenAIChatAutocompletePredictor, OpenAIAutocompletePredictor
from predictors.opensource import (
    stable_code_hf,
    stable_code_vllm,
    deepseek_coder_v2_lite_base,
    codellama_vllm
)

predictor_registry = {
    "openai_gpt_4o_mini_chat_predictor": partial(OpenAIChatAutocompletePredictor, model_name="gpt-4o-mini"),
    "openai_gpt_4o_chat_predictor": partial(OpenAIChatAutocompletePredictor, model_name="gpt-4o"),
    "openai_gpt_3_5_instruct_predictor": partial(OpenAIAutocompletePredictor, model_name="gpt-3.5-turbo-instruct"),
    "stable_code_hf": stable_code_hf,
    "deepseek_coder_v2_lite_base": deepseek_coder_v2_lite_base,
}