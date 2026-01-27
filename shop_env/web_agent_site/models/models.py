"""
Model implementations. The model interface should be suitable for both
the ``site env'' and the ``text env''.
"""
import os
import random
import traceback
from openai import OpenAI
import json
random.seed(4)

class BasePolicy:
    def __init__(self):
        pass

    def forward(observation, available_actions):
        """
        Args:
            observation (`str`):
                HTML string

            available_actions ():
                ...
        Returns:
            action (`str`):
                Return string of the format ``action_name[action_arg]''.
                Examples:
                    - search[white shoes]
                    - click[button=Reviews]
                    - click[button=Buy Now]
        """
        raise NotImplementedError


class HumanPolicy(BasePolicy):
    def __init__(self):
        super().__init__()

    def forward(self, observation, available_actions):
        action = input('> ')
        return action


class RandomPolicy(BasePolicy):
    def __init__(self):
        super().__init__()

    def forward(self, observation, available_actions):
        if available_actions['has_search_bar']:
            action = 'search[shoes]'
        else:
            action_arg = random.choice(available_actions['clickables'])
            action = f'click[{action_arg}]'
        return action

class LLMPolicy(BasePolicy):
    def __init__(self, model_name, mode, model_path=None, device="cuda:0", torch_dtype="bfloat16", api_key=None, base_url=None):
        super().__init__()
        self.call_source = mode
        # API key and base_url should be provided via parameters or environment variables
        openai_api_key = api_key or os.getenv("OPENAI_API_KEY", "{your_api_key}")
        openai_base_url = base_url or os.getenv("OPENAI_BASE_URL", "{your_base_url}")
        
        self.config = {
            "openai":{
                "api_key": openai_api_key,
                "model_name": model_name,
                "max_tokens": 512,
                "temperature": 0.0,
                "base_url": openai_base_url
            }
        }


    def _build_prompt(self, messages):
        prompt = ""
        for message in messages:
            role = message["role"]
            content = message["content"]

            if role == "user":
                prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"
            elif role == "system":
                prompt += f"<|im_start|>system\n{content}<|im_end|>\n"

        prompt += "<|im_start|>assistant\n"
        return prompt

    def get_response_openai(self, messages, config):
        max_try = 50
        cnt = 1
        base_url = config.get('base_url', "{your_base_url}")
        if base_url == "{your_base_url}":
            raise ValueError("Please set IDEALAB_BASE_URL environment variable or provide base_url parameter")
        if config['api_key'] == "{your_api_key}":
            raise ValueError("Please set IDEALAB_API_KEY environment variable or provide api_key parameter")
        client = OpenAI(
            api_key=config['api_key'],
            base_url=base_url,
        )
        while True:
            if cnt > max_try:
                break
            try:
                cnt += 1
                completion = client.chat.completions.create(
                    model=config['model_name'],
                    messages=messages,
                    temperature=0
                )
                return completion.choices[0].message.content
            except Exception as e:
                print(f"LLM 调用失败：{e}")
                traceback.print_exc()
                continue
        return "failed call"

    def get_response_qwen3(self, messages, config):
        base_url = config.get('base_url', "{your_base_url}")
        if base_url == "{your_base_url}":
            raise ValueError("Please set IDEALAB_BASE_URL environment variable or provide base_url parameter")
        if config['api_key'] == "{your_api_key}":
            raise ValueError("Please set IDEALAB_API_KEY environment variable or provide api_key parameter")
        client = OpenAI(
            api_key=config['api_key'],
            base_url=base_url,
        )
        max_try = 60
        cnt = 0

        while True:
            try:
                cnt += 1
                if cnt > max_try:
                    break
                completion = client.chat.completions.create(
                    messages = messages,
                    model=config['model_name'],
                    stream=True,
                    max_tokens=8192,
                    extra_body={
                        "extend_fields": {
                            "chat_template_kwargs": {
                                "enable_thinking": False
                            }
                        }
                    }
                )
                for chunk in completion:
                    if chunk.choices and chunk.choices[0].delta.content and chunk.choices[0].delta.content != '':
                        response = chunk.choices[0].delta.dict().get("content")
                print("*******")
                print(response)
                return response
            except Exception as e:
                print("发生错误：", e)
                continue
        return "failed call"

    def forward(self, observation, env, available_actions):
        if "search" in available_actions['clickables']:
            available_actions['clickables'].remove("search")
        availabel_action_text = f"\n\n搜索功能是否可用: {available_actions['has_search_bar']}\n\n可点击的按钮: {json.dumps(available_actions['clickables'], ensure_ascii=False)}"

        observation = observation + availabel_action_text
        env.history.append({'role': 'user', 'content': observation})

        if self.call_source == 'openai':
            response = self.get_response_openai(env.history, self.config['openai'])
        elif self.call_source == "qwen3":
            response = self.get_response_qwen3(env.history, self.config['openai'])
        else:
            raise ValueError(f"Unsupported call source: {self.call_source}")
        if response == "failed call":
            raise Exception("调用API错误")

        env.history.append({'role': 'assistant', 'content': response})
        #action = response.split("\nAction: ")[1]

        return response
