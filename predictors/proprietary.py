from predictors.base import AutocompletePredictor

import openai

openai.api_key = open(".open-ai-api-key").read().strip()

OPENAI_PRICING = {
    "gpt-4o-mini": {
        "input": 0.15 / 1_000_000,
        "output": 0.6 / 1_000_000
    },
    "gpt-4o": {
        "input": 5 / 1_000_000,
        "output": 15 / 1_000_000
    },
    "gpt-3.5-turbo-instruct": {
        "input": 1.5 / 1_000_000,
        "output": 2 / 1_000_000
    }
}

DEFAULT_GPT_SYSTEM_PROMPT = """
You are a YAML autocomplete system. You are provided with file contents in the following format:

Code after the cursor:
<code>

Code before the cursor:
<code>

Your task is to create an autocompletion until the end of the line for this file. Only output the actual code for the completion.

Examples:

Code after the cursor:
  - python=3.9
  - gh
  - pip
  - ipykernel
  - pip:
    - github3.py
    - tqdm
    
Code before the cursor:
name: yaml-complete
channels:
  - conda-forge
  - defaults
  
Completion:
dependencies:
"""

DEFAULT_GPT_USER_PROMPT = """
Code after the cursor:
{text_after}

Code before the cursor:
{text_before}

Completion: 
"""

class OpenAIChatAutocompletePredictor(AutocompletePredictor):
    def __init__(
        self, 
        system_prompt: str = DEFAULT_GPT_SYSTEM_PROMPT, 
        user_prompt: str = DEFAULT_GPT_USER_PROMPT,
        model_name: str = "gpt-4o-mini"
    ):
        self.model_name = model_name
        self.user_prompt = user_prompt
        self.system_prompt = system_prompt
        self.total_cost = 0

    def predict(self, text_before: str, text_after: str) -> str:
        chat_completion = openai.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.user_prompt.format(
                    text_after=text_after,
                    text_before=text_before
                )}
            ]
        )
        self.total_cost += chat_completion.usage.completion_tokens * OPENAI_PRICING[self.model_name]['output'] + \
            chat_completion.usage.prompt_tokens * OPENAI_PRICING[self.model_name]['input']
        return chat_completion.choices[0].message.content.split("\n")[0]
    
    def flush_cost(self):
        self.total_cost = 0

DEFAULT_GPT_INSTRUCT_PROMPT = """
# Task formulation:
You are a YAML autocomplete system. You are provided with file contents in the following format:

Code after the cursor:
<code>

Code before the cursor:
<code>

Your task is to create an autocompletion until the end of the line for this file. Only output the actual code for the completion.

# Examples:

Code after the cursor:
  - python=3.9
  - gh
  - pip
  - ipykernel
  - pip:
    - github3.py
    - tqdm
    
Code before the cursor:
name: yaml-complete
channels:
  - conda-forge
  - defaults
  
Completion:
dependencies:

# Current completion

Code after the cursor:
{text_after}

Code before the cursor:
{text_before}

Completion: 
"""

class OpenAIAutocompletePredictor(AutocompletePredictor):
    def __init__(
        self, 
        prompt: str = DEFAULT_GPT_INSTRUCT_PROMPT, 
        model_name: str = "gpt-3.5-turbo-instruct"
    ):
        self.model_name = model_name
        self.prompt = prompt
        self.total_cost = 0

    def predict(self, text_before: str, text_after: str) -> str:
        completion = openai.completions.create(
            model=self.model_name,
            prompt=DEFAULT_GPT_INSTRUCT_PROMPT.format(
                text_after=text_after,
                text_before=text_before
            )
        )
        self.total_cost += completion.usage.completion_tokens * OPENAI_PRICING[self.model_name]['output'] + \
            completion.usage.prompt_tokens * OPENAI_PRICING[self.model_name]['input']
        return completion.choices[0].text.split("\n")[0]

    def flush_cost(self):
        self.total_cost = 0