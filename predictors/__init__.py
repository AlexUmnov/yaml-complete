from predictors.proprietary import OpenAIChatAutocompletePredictor, OpenAIAutocompletePredictor
from predictors.opensource import stable_code_hf, codegen_350_hf

predictor_registry = {
    "openai_gpt_4o_mini_chat_predictor": OpenAIChatAutocompletePredictor(model_name="gpt-4o-mini"),
    "openai_gpt_4o_chat_predictor": OpenAIChatAutocompletePredictor(model_name="gpt-4o"),
    "openai_gpt_3_5_instruct_predictor": OpenAIAutocompletePredictor(model_name="gpt-3.5-turbo-instruct"),
#    "stable_code_hf": stable_code_hf(),
    "codegen_350_hf": codegen_350_hf()
}