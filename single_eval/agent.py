"""Agent module: implements main logic for shopping agent."""

import argparse
import json
import os
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import yaml
from openai import OpenAI
from tqdm import tqdm

from env import ShopEnv

DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_RETRY = 200
IDEALAB_DEFAULT_KEY = "{your_api_key}"  # Should be set via config file or environment variable
IDEALAB_DEFAULT_BASE_URL = "{your_base_url}"  # Should be set via config file or environment variable
FAILED_CALL_MESSAGE = "failed call"


class Agent:
    """Shopping agent class responsible for interacting with shopping environment."""

    def __init__(self, task_id: int, config: Dict[str, Any]) -> None:
        """
        Initialize Agent.

        Args:
            task_id: Task ID
            config: Dictionary containing model config, API config, etc.
        """
        self.task_id = task_id
        self.config = config
        self.model_name = self.config["model_name"]
        self.source = config["source"]

        if self.source == "idealab":
            default_key = IDEALAB_DEFAULT_KEY
            default_base_url = IDEALAB_DEFAULT_BASE_URL
        else:
            default_key = None
            default_base_url = None

        self.model_key = self.config.get("model_key", default_key)
        self.base_url = self.config.get("base_url", default_base_url)

        api_config = {
            "idealab": {
                "api_key": self.model_key,
                "model_name": self.model_name,
                "max_tokens": DEFAULT_MAX_TOKENS,
                "temperature": DEFAULT_TEMPERATURE,
            }
        }
        if self.source not in api_config:
            raise ValueError(f"Unsupported data source: {self.source}")

        self.api_config = api_config[self.source]
        self.system_prompt = config["system_prompt"]

        self.shop_env: Optional[ShopEnv] = None
        self.task_complete = False
        self.conversation_log: List[Dict[str, Any]] = []
        self.messages: List[Dict[str, str]] = []
        self.user_persona: Optional[Dict[str, Any]] = None
        self.env_idx: Optional[int] = None

    def set_shop_env(self, shop_env: ShopEnv) -> None:
        """
        Set ShopEnv instance.

        Args:
            shop_env: ShopEnv instance
        """
        self.shop_env = shop_env

    def reset(self) -> str:
        """
        Reset state and start new task.

        Returns:
            str: Initial instruction

        Raises:
            AttributeError: When shop_env is not initialized
        """
        if self.shop_env is None:
            raise AttributeError("self.shop_env not initialized")

        env_result = self.shop_env.reset(self.task_id)
        self.env_idx = self.shop_env.env_idx
        instruction = env_result["instruction"] + "\n\n搜索功能是否可用: True\n\n可点击的按钮: []"

        self.task_complete = False
        self.conversation_log = []

        if self.shop_env.if_persona and "user_persona" in env_result:
            self.user_persona = env_result["user_persona"]
            persona_str = json.dumps(self.user_persona, ensure_ascii=False)
            persona_content = f"\n用户的个人文档是：{persona_str}"
            self.messages = [
                {"role": "system", "content": self.system_prompt + persona_content}
            ]
        else:
            self.messages = [{"role": "system", "content": self.system_prompt}]

        return instruction

    def act(self, instruction: str) -> Tuple[bool, str]:
        """
        Execute one interaction step.

        Args:
            instruction: Instruction content

        Returns:
            Tuple[bool, str]: (task_complete, observation)
                - task_complete: Whether task is complete
                - observation: Observation result

        Raises:
            AttributeError: When shop_env is not initialized
            ValueError: When environment error occurs
            RuntimeError: When LLM call fails
        """
        if self.shop_env is None:
            raise AttributeError(
                "Agent not correctly connected to ShopEnv, please check the configuration."
            )

        self.messages.append({"role": "user", "content": instruction})

        llm_response = self.get_response_idealab(self.messages)
        llm_response = self._clean_response(llm_response)

        if llm_response == FAILED_CALL_MESSAGE:
            raise RuntimeError("LLM call failed")

        self.messages.append({"role": "assistant", "content": llm_response})
        action = llm_response

        env_response = self.shop_env.interact(action)
        if "error" in env_response:
            print(action)
            raise ValueError(f"环境报错: {env_response['error']}")

        observation = env_response.get("instruction", "")

        if env_response.get("done", False) or env_response.get("over", False):
            reward = env_response.get("reward", 0)
            reward_detail = env_response.get("reward_detail", {})
            goal = env_response.get("goal", {})
            purchase = env_response.get("purchase", {})

            self.save_to_json(reward, reward_detail, goal, purchase)
            self.task_complete = True

        return self.task_complete, observation

    def _clean_response(self, response: str) -> str:
        """
        Clean LLM response by removing redacted_reasoning tags.

        Args:
            response: Raw response

        Returns:
            str: Cleaned response
        """
        if "</think>" in response:
            response = response.split("</think>")[-1].strip("\n")
        return response

    def get_response_idealab(
        self, messages: List[Dict[str, str]], max_try: int = DEFAULT_MAX_RETRY
    ) -> str:
        """
        Call idealab LLM API.

        Args:
            messages: Message list
            max_try: Maximum retry attempts

        Returns:
            str: LLM response content, returns "failed call" on failure
        """
        if self.model_key is None:
            raise ValueError("model_key not set, cannot call API")
        if self.base_url is None:
            raise ValueError("base_url not set, cannot call API")

        client = OpenAI(
            api_key=self.model_key,
            base_url=self.base_url,
        )

        for attempt in range(max_try):
            try:
                completion = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=DEFAULT_TEMPERATURE,
                )
                return completion.choices[0].message.content
            except Exception as e:
                print(f"LLM call failed (attempt {attempt + 1}/{max_try}): {e}")
                if attempt == max_try - 1:
                    traceback.print_exc()
                continue

        return FAILED_CALL_MESSAGE

    def save_to_json(
        self,
        reward: float,
        reward_detail: Dict[str, Any],
        goal: Dict[str, Any],
        purchase: Dict[str, Any],
    ) -> None:
        """
        Save conversation and scores to JSON file.

        Args:
            reward: Reward score
            reward_detail: Reward details
            goal: Goal information
            purchase: Purchase information
        """
        log_data = {
            "task_id": self.task_id,
            "reward": reward,
            "reward_detail": reward_detail,
            "goal": goal,
            "purchase": purchase,
            "conversation": self.messages,
        }

        file_path = os.path.join(
            self.config["output_path"], self.config["model_name"]
        )
        os.makedirs(file_path, exist_ok=True)

        filename = os.path.join(file_path, f"{self.task_id}.json")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=4)

        print(f"[LOG] Saved to file: {filename}")


def get_finished_task(out_path: str) -> List[int]:
    """
    Get list of completed tasks.

    Args:
        out_path: Output directory path

    Returns:
        List[int]: List of completed task IDs
    """
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)

    json_files = []
    for filename in os.listdir(out_path):
        if filename.endswith(".json"):
            task_id_str = filename.rsplit(".", 1)[0]
            try:
                json_files.append(int(task_id_str))
            except ValueError:
                print(
                    f"Warning: Unable to parse task ID '{task_id_str}', skipping file"
                )

    return sorted(json_files)


def run_task(task_id: int, config: Dict[str, Any]) -> None:
    """
    Run single task.

    Args:
        task_id: Task ID
        config: Configuration dictionary
    """
    agent: Optional[Agent] = None
    shop_env: Optional[ShopEnv] = None

    try:
        shop_env = ShopEnv(config["env_config"])
        agent = Agent(task_id=task_id, config=config["agent_config"])
        agent.set_shop_env(shop_env)
        observation = agent.reset()

        while True:
            done, observation = agent.act(observation)
            if done:
                break
    except Exception as e:
        print(f"Error (task {task_id}): {e}")
        traceback.print_exc()
    finally:
        if agent is not None and agent.shop_env is not None:
            agent.shop_env.release()


def run_tasks_multithreaded(
    todo_tasks: List[int], config: Dict[str, Any], max_workers: int = 4
) -> None:
    """
    Execute tasks using multithreading.

    Args:
        todo_tasks: List of task IDs to process
        config: Configuration dictionary
        max_workers: Maximum number of worker threads
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(run_task, task_id, config): task_id
            for task_id in todo_tasks
        }

        for future in tqdm(
            as_completed(future_to_task),
            total=len(todo_tasks),
            desc="执行任务",
        ):
            task_id = future_to_task[future]
            try:
                future.result()
                print(f"**Task {task_id} completed**")
            except Exception as e:
                print(f"**Task {task_id} failed**: {e}")


def run_tasks_singlethreaded(
    todo_tasks: List[int], config: Dict[str, Any]
) -> None:
    """
    Execute tasks in single-threaded mode.

    Args:
        todo_tasks: List of task IDs to process
        config: Configuration dictionary
    """
    for task_id in tqdm(todo_tasks, desc="执行任务"):
        print(f"**Begin task {task_id}**")
        run_task(task_id, config)


def main() -> None:
    """Main function: parse arguments and execute tasks."""
    parser = argparse.ArgumentParser(
        description="Batch process tasks from YAML config files."
    )
    parser.add_argument(
        "--yaml_name", required=True, help="YAML config file name"
    )
    parser.add_argument(
        "--multithread",
        action="store_true",
        help="Whether to use multithreading to execute",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Maximum number of worker threads in multithreaded mode",
    )
    args = parser.parse_args()

    config_file = args.yaml_name
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found '{config_file}'")
        return
    except yaml.YAMLError as e:
        print(f"Error: YAML parsing failed: {e}")
        return

    agent_config = config.get("agent_config")
    if agent_config is None:
        print("Error: 'agent_config' missing in config file")
        return

    output_path = os.path.join(
        agent_config["output_path"], agent_config["model_name"]
    )
    finished_tasks = get_finished_task(output_path)
    all_tasks = list(range(agent_config.get("task_nums", 0)))
    todo_tasks = [i for i in all_tasks if i not in finished_tasks]

    print(
        f"Total tasks: {len(all_tasks)}, "
        f"Completed: {len(finished_tasks)}, "
        f"Pending: {len(todo_tasks)}"
    )

    if args.multithread:
        print(f"Using multithreaded mode, max workers: {args.max_workers}")
        run_tasks_multithreaded(todo_tasks, config, args.max_workers)
    else:
        print("Using single-threaded mode")
        run_tasks_singlethreaded(todo_tasks, config)


if __name__ == "__main__":
    main()
