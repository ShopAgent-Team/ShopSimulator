"""Shopper module: implements main logic for shopper simulator."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import pdb
from openai import OpenAI

logger = logging.getLogger(__name__)

DEFAULT_MAX_RETRY = 50
DEFAULT_MODEL_NAME = "qwen3-235b-a22b-instruct-2507"
DEFAULT_SOURCE = "openai"
IDEALAB_DEFAULT_KEY = "{your_api_key}"  # Should be set via config file or environment variable
IDEALAB_DEFAULT_BASE_URL = "{your_base_url}"  # Should be set via config file or environment variable
DEFAULT_DATA_FILE = "../ShopEnv/data/fine_items_eval_train_all.json"
FAILED_CALL_MESSAGE = "failed call"

# 系统提示词
SYSTEM_PROMPT = """你是一个模拟的购物者（shopper），正在尝试通过与客服代理（agent）的对话完成一次商品购买任务。你的任务是：

1. 你有一个**具体的购买目标**（只有你知道），但你不会一开始就透露它。你需要从一个模糊的购买目标出发，与 agent 进行自然的多轮对话。
2. 在对话中，你主要作用是回答 agent 的提问，帮助它逐渐了解你的购买需求，最终让 agent 帮你找到并推荐出你实际想买的商品。
3. 注意：**不要主动提供详细信息，而是等待agent向你询问。**
5. 注意：**在对话结束前，也就是agent准备要购买之前，确保agent已经得到了具体购买目标中的所有特征和属性，不可以遗漏任何信息。**
6. 若有信息未告知agent，请拒绝agent的购买行为，并提供一个简单的理由，等待agent的更多询问。

请保持语言自然、口语化，像一个真实的人在购物时那样思考和表达。
"""


class ShopperSimulator:
    """Shopper simulator class for simulating real shopper conversation behavior."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize shopper simulator.

        Args:
            config: Configuration dictionary containing:
                - model_name: Model name (optional)
                - source: Data source (optional, default 'openai')
                - model_key: API key (optional)
                - base_url: API base URL (optional)
                - system_prompt: Custom system prompt (optional)
                - data_file: Product data file path (optional)

        Raises:
            FileNotFoundError: When product data file cannot be found
            ValueError: When configuration parameters are invalid
        """
        self.config = config
        self.messages: List[Dict[str, str]] = []
        self.max_retry = DEFAULT_MAX_RETRY
        self.model_name = config.get("model_name") or DEFAULT_MODEL_NAME
        self.source = config.get("source", DEFAULT_SOURCE)

        if self.source == "openai":
            default_key = IDEALAB_DEFAULT_KEY
            default_base_url = IDEALAB_DEFAULT_BASE_URL
        else:
            default_key = None
            default_base_url = None

        self.model_key = config.get("model_key", default_key)
        self.base_url = config.get("base_url", default_base_url)

        system_prompt_config = config.get("system_prompt", "")
        self.base_system_prompt = system_prompt_config if system_prompt_config else SYSTEM_PROMPT
        self.system_prompt = self.base_system_prompt


    def _get_response_openai(self, messages: List[Dict[str, str]], max_retry: int = None) -> str:
        """
        Get LLM response via OpenAI API.

        Args:
            messages: Conversation message list
            max_retry: Maximum retry attempts, defaults to instance's max_retry

        Returns:
            str: LLM generated response content, returns FAILED_CALL_MESSAGE on failure

        Raises:
            ValueError: When model_key or base_url is not set
        """
        if self.model_key is None:
            raise ValueError("model_key not set, cannot call API")
        if self.base_url is None:
            raise ValueError("base_url not set, cannot call API")

        client = OpenAI(api_key=self.model_key, base_url=self.base_url)
        max_retry = max_retry or self.max_retry

        for attempt in range(max_retry):
            try:
                completion = client.chat.completions.create(
                    model=self.model_name, messages=messages
                )
                return completion.choices[0].message.content
            except Exception as e:
                logger.warning(f"LLM call failed (attempt {attempt + 1}/{max_retry}): {e}")
                if attempt == max_retry - 1:
                    logger.error(f"LLM call finally failed after {max_retry} retries")
                    return FAILED_CALL_MESSAGE
                continue

        return FAILED_CALL_MESSAGE

    def step(self, agent_response: str) -> str:
        """
        Receive Agent's response and generate Shopper's response.

        Args:
            agent_response: Agent's response content

        Returns:
            str: Shopper's response content

        Raises:
            ValueError: When unsupported data source
        """
        self.messages.append({"role": "user", "content": agent_response})

        if self.source == "openai":
            shopper_response = self._get_response_openai(self.messages)
        else:
            raise ValueError(f"Unsupported source: {self.source}")

        self.messages.append({"role": "assistant", "content": shopper_response})

        return shopper_response

    def reset(self, instruction: str, goal_options: List[str]) -> None:
        """
        Reset Shopper state for new task.

        Args:
            task_id: Task ID used to retrieve corresponding task information from product_data

        Raises:
            KeyError: When task_id does not exist in product_data
            IndexError: When instructions list is empty
        """

        self.instruction = instruction
        self.options = json.dumps(goal_options, ensure_ascii=False)
        self.messages = []

        # 构建任务信息并更新系统提示词
        task_info = (
            f"\n\n你本轮的购买任务是: {self.instruction} \n"
            f"你的规格选择是: {self.options}"
        )
        self.system_prompt = self.base_system_prompt + task_info
        self.messages.append({"role": "system", "content": self.system_prompt})