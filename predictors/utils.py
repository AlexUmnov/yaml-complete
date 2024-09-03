from typing import Optional

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

def quantize_model_autoawq(model_name:str, out_path, tokenizer_name: Optional[str] = None):
    quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

    # Load model
    model = AutoAWQForCausalLM.from_pretrained(
        model_name, **{"low_cpu_mem_usage": True, "use_cache": False}
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name if tokenizer_name else model_name, trust_remote_code=True)

    # Quantize
    model.quantize(tokenizer, quant_config=quant_config)

    # Save quantized model
    model.save_quantized(out_path)
    tokenizer.save_pretrained(out_path)

    print(f'Model is quantized and saved at "{out_path}"')