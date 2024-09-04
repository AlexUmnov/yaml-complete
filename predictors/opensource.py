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
        device: Optional[str] = None
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name if tokenizer_name else model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **init_params 
        )
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else self.tokenizer.eos_token_id

        self.device = device
        if device:
            self.model.to(device)

        self.inference_params = inference_params
        self.inferencing_prompt = inferencing_prompt
        self.total_cost = 0
        

    def predict(self, text_before: str, text_after: str) -> str:
        prompt = self.inferencing_prompt.format(
            text_after=text_after,
            text_before=text_before
        )
        input_ids = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)["input_ids"]

        outputs = self.model.generate(
            input_ids, 
            tokenizer=self.tokenizer, 
            **self.inference_params
        )

        result = self.tokenizer.decode(outputs[0])

        if prompt in result:
            result = result.split(prompt)[1]
        result = result.strip(self.tokenizer.eos_token)
        result = result.strip(self.tokenizer.pad_token)
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

deepseek_coder_v2_lite_base_hf = partial(HuggingFacePredictor, 
    model_name="deepseek-ai/DeepSeek-Coder-V2-Lite-Base",
    init_params={
        "device_map": "auto",
        "trust_remote_code": True,
        "torch_dtype": torch.float16,
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

codegen_350_mono_hf = partial(HuggingFacePredictor, 
    model_name="Salesforce/codegen-350M-mono",
    tokenizer_name="Salesforce/codegen-350M-mono",
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

yaml_complete_fintetuned = partial(HuggingFacePredictor, 
    model_name="alexvumnov/yaml_completion",
    tokenizer_name="alexvumnov/yaml_completion",
    init_params={
        "device_map": "auto"
    },
    inference_params={
        "max_new_tokens": 128,
        "stop_strings": ["\n", "<|endoftext|>"]
    },
    inferencing_prompt= """
# Here's a yaml file to offer a completion for
# Lines after the current one
{text_after}
# Lines before the current one
{text_before}
# Completion:
"""
)

yaml_complete_fintetuned_8_bit = partial(HuggingFacePredictor, 
    model_name="alexvumnov/yaml_completion_8bit",
    tokenizer_name="alexvumnov/yaml_completion_8bit",
    init_params={
        "device_map": "auto"
    },
    inference_params={
        "max_new_tokens": 128,
        "stop_strings": ["\n", "<|endoftext|>"]
    },
    inferencing_prompt= """
# Here's a yaml file to offer a completion for
# Lines after the current one
{text_after}
# Lines before the current one
{text_before}
# Completion:
"""
)

yaml_complete_fintetuned_cpu = partial(HuggingFacePredictor, 
    model_name="alexvumnov/yaml_completion",
    tokenizer_name="alexvumnov/yaml_completion",
    init_params={},
    device='cpu',
    inference_params={
        "max_new_tokens": 128,
        "stop_strings": ["\n", "<|endoftext|>"]
    },
    inferencing_prompt= """
# Here's a yaml file to offer a completion for
# Lines after the current one
{text_after}
# Lines before the current one
{text_before}
# Completion:
"""
)

codegen2_16_b_p = partial(HuggingFacePredictor, 
    model_name="Salesforce/codegen2-16B_P",
    tokenizer_name="Salesforce/codegen2-16B_P",
    init_params={
        "device_map": "auto",
        "torch_dtype": torch.float16,
    },
    inference_params={
        "max_new_tokens": 128,
        "stop_strings": ["\n"]
    },
    inferencing_prompt="""
{text_bofore} + "<mask_1>" + {text_after} + "<|endoftext|>" + "<sep>" + "<mask_1>"
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