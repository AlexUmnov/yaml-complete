from typing import Dict, Optional, Any
from functools import partial

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from vllm import LLM, SamplingParams

from predictors.base import AutocompletePredictor


class HuggingFacePredictor(AutocompletePredictor):
    def __init__(
        self,
        model_name: str,
        inferencing_prompt: str,
        tokenizer_name: Optional[str] = None,
        init_params: Dict[str, Any] = {},
        inference_params: Dict[str, Any] = {},
    ):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **init_params 
        )

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name if tokenizer_name else model_name)

        self.inference_params = inference_params
        self.inferencing_prompt = inferencing_prompt
        self.total_cost = 0
        

    def predict(self, text_before: str, text_after: str) -> str:
        prompt = self.inferencing_prompt.format(
            text_after=text_after,
            text_before=text_before
        )
        input_ids = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)["input_ids"]

        outputs = self.model.generate(input_ids, tokenizer=self.tokenizer, **self.inference_params)

        result = self.tokenizer.decode(outputs[0])

        if prompt in result:
            result = result.split(prompt)[1]
        return result

stable_code_hf = partial(HuggingFacePredictor, 
    model_name="stabilityai/stable-code-3b",
    tokenizer_name="stabilityai/stable-code-3b",
    init_params={
        "device_map": "auto"
    },
    inference_params={
        "max_new_tokens": 128,
        "stop_strings": ["\n"]
    },
    inferencing_prompt="""```yaml
    # Text before
    {text_after}
    # Complete Yaml:
    {text_before}"""
)

deepseek_coder_v2_lite_base = partial(HuggingFacePredictor, 
    model_name="deepseek-ai/DeepSeek-Coder-V2-Lite-Base",
    init_params={
        "device_map": "auto"
    },
    inference_params={
        "max_new_tokens": 128,
        "stop_strings": ["\n"]
    },
    inferencing_prompt="""```yaml
    # Text before
    {text_after}
    # Complete Yaml:
    {text_before}"""
)

codellama_hf = partial(HuggingFacePredictor, 
    model_name="codellama/CodeLlama-7b-hf",
    init_params={
        "device_map": "auto"
    },
    inference_params={
        "max_new_tokens": 128,
        "stop_strings": ["\n"]
    },
    inferencing_prompt="""```yaml
    # Text before
    {text_after}
    # Complete Yaml:
    {text_before}"""
)

codegen_350_hf = partial(HuggingFacePredictor, 
    model_name="Salesforce/codegen-350M-multi",
    tokenizer_name="Salesforce/codegen-350M-multi",
    init_params={
        "device_map": "auto"
    },
    inference_params={
        "max_new_tokens": 128,
        "stop_strings": ["\n"]
    },
    inferencing_prompt="""
#file.yaml
{text_before}
"""
)

class VLLMPredictor(AutocompletePredictor):
    def __init__(
        self,
        model_name: str,
        inferencing_prompt: str,
        init_params: Dict[str, Any] = {},
        tokenizer_name: Optional[str] = None,
        inference_params: Dict[str, Any] = {},
    ):
        self.model = LLM(
            model=model_name,
            tokenizer=tokenizer_name if tokenizer_name else model_name, 
            **init_params
        )
        self.inference_params = SamplingParams(**inference_params)
        self.inferencing_prompt = inferencing_prompt
        self.total_cost = 0
    
    def predict(self, text_before: str, text_after: str) -> str:
        prompt = self.inferencing_prompt.format(
            text_after=text_after,
            text_before=text_before
        )

        outputs = self.model.generate([prompt], self.inference_params)

        return outputs[0].outputs[0].text.split("\n")[0]

        
stable_code_vllm = partial(VLLMPredictor, 
    model_name="stabilityai/stable-code-3b",
    init_params={
        "max_model_len": 1024
    },
    inference_params={
        "n": 1,
        "max_tokens": 128,
    },
    inferencing_prompt="""```yaml
    # Text before
    {text_after}
    # Complete Yaml:
    {text_before}"""
)