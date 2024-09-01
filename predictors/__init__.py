from predictors.proprietary import OpenAIChatAutocompletePredictor, OpenAIAutocompletePredictor

predictor_registry = {
    "openai_gpt_4o_mini_chat_predictor": OpenAIChatAutocompletePredictor(model_name="gpt-4o-mini"),
    "openai_gpt_4o_chat_predictor": OpenAIChatAutocompletePredictor(model_name="gpt-4o"),
    "openai_gpt_3_5_instruct_predictor": OpenAIAutocompletePredictor(model_name="gpt-3.5-turbo-instruct"),
}