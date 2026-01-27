"""Agent module: implements main logic for shopping agent."""

import argparse
import json
import os
import re
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import yaml
from openai import OpenAI
from tqdm import tqdm

from env import ShopEnv
from shopper import ShopperSimulator

import pdb
DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_RETRY = 50
IDEALAB_DEFAULT_KEY = "{your_api_key}"  # Should be set via config file or environment variable
IDEALAB_DEFAULT_BASE_URL = "{your_base_url}"  # Should be set via config file or environment variable

SEP_MARKER = "[SEP]"
ACTION_TYPE_INTERACT = "interact_with_env"
ACTION_TYPE_SHOPPER = "ask_shopper"
ROLE_ENV = "Env"
ROLE_SHOPPER = "Shopper"
FAILED_CALL_MESSAGE = "failed call"

class Agent:
    """Shopping agent class responsible for interacting with shopping environment and shoppers."""

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
        self.agent_format_template = config.get("agent_format_template", "")

        self.shopper_simulator: Optional[ShopperSimulator] = None
        self.shop_env: Optional[ShopEnv] = None
        self.task_complete = False
        self.shopping_history: List[Dict[str, Any]] = []
        self.conversation_log: List[Dict[str, Any]] = []
        self.action_list: List[str] = []
        self.obs_list: List[str] = []
        self.messages: List[Dict[str, str]] = []
        self.turn: int = 0
        self.max_turns: int = 0
        self.search_available: bool = True
        self.available_buttons: List[str] = []
        self.user_persona: Optional[Dict[str, Any]] = None
        self.env_idx: Optional[int] = None

    def set_shopper(self, shopper_simulator: ShopperSimulator) -> None:
        """
        Set Shopper instance.

        Args:
            shopper_simulator: ShopperSimulator instance
        """
        self.shopper_simulator = shopper_simulator

    def set_shop_env(self, shop_env: ShopEnv) -> None:
        """
        Set ShopEnv instance.

        Args:
            shop_env: ShopEnv instance
        """
        self.shop_env = shop_env

    def reset(self) -> Tuple[str, str]:
        """
        Reset state and start new task.

        Returns:
            Tuple[str, str]: (instruction, shopper_input) instruction and shopper input

        Raises:
            AttributeError: When shop_env or shopper_simulator is not initialized
        """
        if self.shop_env is None:
            raise AttributeError("self.shop_env not initialized")

        env_result = self.shop_env.reset(self.task_id)
        self.env_idx = self.shop_env.env_idx
        instruction = env_result["instruction"] + "\n\n搜索功能是否可用: True\n\n可点击的按钮: []"
        goal_options = env_result["goal_options"]

        if self.shopper_simulator is None:
            raise AttributeError("self.shopper_simulator not initialized")

        self.shopper_simulator.reset(instruction, goal_options)
        self.task_complete = False
        self.shopping_history = []
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

        shopper_input = self._get_shopper_input()
        if not shopper_input:
            raise AttributeError("shopper response error")

        return instruction, shopper_input

    def parse_output(self, llm_response: str) -> Dict[str, Optional[str]]:
        """
        Parse LLM response output.

        Args:
            llm_response: Raw response string from LLM

        Returns:
            Dict[str, Optional[str]]: Dictionary containing thought, action_type, action_content
        """
        result = {
            "thought": None,
            "action_type": None,
            "action_content": None,
        }

        for line in llm_response.strip().split("\n"):
            line = line.strip()
            if not line:
                continue

            if line.startswith("Thought:") or line.startswith("Thought："):
                result["thought"] = line.split(":", 1)[-1].strip()

            elif line.startswith("Action_type:") or line.startswith("Action_type："):
                result["action_type"] = line.split(":", 1)[-1].strip()

            elif line.startswith("Action_content:") or line.startswith("Action_content："):
                result["action_content"] = line.split(":", 1)[-1].strip()

        return result

    def process_observation(self, observation: str) -> str:
        """
        Process observation, extract content after second [SEP].

        Args:
            observation: Raw observation string

        Returns:
            str: Processed observation string
        """
        instruction_prefix = "Instruction: [SEP]"
        first_sep = observation.find(instruction_prefix)

        if first_sep == -1:
            return observation

        second_sep = observation.find(SEP_MARKER, first_sep + len(instruction_prefix))

        if second_sep == -1:
            return observation

        result = SEP_MARKER + observation[second_sep + len(SEP_MARKER):]
        return result

    def parse_instruction_in_envobs(
        self, instruction: str
    ) -> Tuple[str, str, List[str]]:
        """
        Parse instruction information in environment observation.

        Args:
            instruction: String containing instruction and page state

        Returns:
            Tuple[str, str, List[str]]: (observation, search_available, buttons)
                - observation: Page state
                - search_available: Whether search is available ("True" or "False")
                - buttons: List of clickable buttons
        """
        observation = instruction.split("搜索功能是否可用:")[0].strip()

        search_pattern = r"搜索功能是否可用: (True|False)"
        search_match = re.search(search_pattern, instruction)
        search_available = search_match.group(1) if search_match else "True"

        button_pattern = r"可点击的按钮:\s*\[(.*?)\]"
        button_match = re.search(button_pattern, instruction)
        if button_match:
            buttons_str = button_match.group(1)
            buttons = [btn.strip().strip('"') for btn in buttons_str.split(",") if btn.strip()]
        else:
            buttons = []

        return observation, search_available, buttons

    def act(
        self, user_role: str, instruction: str
    ) -> Tuple[bool, str, str]:
        """
        Execute one interaction step.

        Args:
            user_role: User role ("ENV" or "Shopper")
            instruction: Instruction content

        Returns:
            Tuple[bool, str, str]: (task_complete, role, observation)
                - task_complete: Whether task is complete
                - role: Next turn's role
                - observation: Observation result

        Raises:
            RuntimeError: When LLM call fails
            ValueError: When user_role is wrong or environment error occurs
        """
        if self.shopper_simulator is None or self.shop_env is None:
            raise AttributeError("Agent not correctly connected to Shopper or ShopEnv, please check the configuration.")

        prompt = f"{user_role}: {instruction}"

        if user_role == ROLE_ENV:
            observation = prompt
            self.obs_list.append(observation)
            prompt += f"\n轮次: {self.turn}\n请做出下一轮决策:"
        elif user_role == ROLE_SHOPPER:
            prompt += f"\n轮次: {self.turn}\n请做出下一轮决策:"
        else:
            raise ValueError(f"user_role error: {user_role}")

        self.messages.append({"role": "user", "content": prompt})

        if self.source == "idealab":
            llm_response = self.get_response_idealab(self.messages)
        else:
            raise ValueError(f"Unsupported data source: {self.source}")

        llm_response_raw = llm_response
        llm_response = self._clean_response(llm_response)

        if llm_response == FAILED_CALL_MESSAGE:
            raise RuntimeError("LLM call failed")

        try:
            response_dict = self.parse_output(llm_response)
        except Exception as e:
            self.messages.append({"role": "assistant", "content": llm_response_raw})
            print(f"Action parsing failed: {e}")
            return False, ROLE_ENV, "Action解析失败，请重新决策。"

        action = response_dict["action_type"]
        content = response_dict["action_content"]
        self.messages.append({"role": "assistant", "content": llm_response})

        if action == ACTION_TYPE_INTERACT:
            return self._handle_env_interaction(content)
        elif action == ACTION_TYPE_SHOPPER:
            if self.shopper_simulator is None:
                raise AttributeError("shopper_simulator not initialized")
            observation = self.shopper_simulator.step(content)
            return self.task_complete, ROLE_SHOPPER, observation
        else:
            raise ValueError(f"Unsupported action type: {action}")
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
        if "<think>\n\n" in response:
            response = response.split("<think>\n\n")[-1].strip("\n")
        return response

    def _handle_env_interaction(self, content: str) -> Tuple[bool, str, str]:
        """
        Handle environment interaction operation.

        Args:
            content: Operation content

        Returns:
            Tuple[bool, str, str]: (task_complete, role, observation)
        """
        if self.shop_env is None:
            raise AttributeError("shop_env not initialized")

        role = ROLE_ENV
        env_response = self.shop_env.interact(f"\nAction: {content}")

        self.action_list.append(content)

        if "error" in env_response:
            error_msg = env_response["error"]
            print(f"Environment error: {error_msg}")
            raise ValueError(f"Environment error: {error_msg}")

        observation = env_response["instruction"]
        observation = self.process_observation(observation)

        if env_response.get("done", False) or env_response.get("over", False):
            reward = env_response.get("reward", 0)
            reward_detail = env_response.get("reward_detail", {})
            goal = env_response.get("goal", {})
            purchase = env_response.get("purchase", {})

            if self.shopper_simulator is None:
                raise AttributeError("shopper_simulator not initialized")

            self.save_to_json(
                reward, reward_detail, goal, purchase, self.shopper_simulator.messages
            )
            self.task_complete = True

        return self.task_complete, role, observation

    def _get_shopper_input(self) -> str:
        """
        Get latest reply from Shopper.

        Returns:
            str: Shopper's reply content

        Raises:
            AttributeError: When shopper_simulator is not initialized
        """
        if self.shopper_simulator is None:
            raise AttributeError("shopper_simulator not initialized")

        shopper_output = self.shopper_simulator.step("请提供您的购物需求。")
        if shopper_output:
            self.conversation_log.append({"shopper": shopper_output})
        return shopper_output

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
        messages: List[Dict[str, str]],
    ) -> None:
        """
        Save conversation and scores to JSON file.

        Args:
            reward: Reward score
            reward_detail: Reward details
            goal: Goal information
            purchase: Purchase information
            messages: Shopper conversation messages
        """
        log_data = {
            "task_id": self.task_id,
            "reward": reward,
            "reward_detail": reward_detail,
            "goal": goal,
            "purchase": purchase,
            "conversation": self.messages,
            "shopper_conversation": messages,
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
                print(f"Warning: Unable to parse task ID '{task_id_str}', skipping file")

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
        shopper = ShopperSimulator(config["shopper_config"])
        shop_env = ShopEnv(config["env_config"])

        agent = Agent(task_id=task_id, config=config["agent_config"])

        agent.set_shopper(shopper)
        agent.set_shop_env(shop_env)
        task_description, observation = agent.reset()

        role = ROLE_SHOPPER
        agent.turn = 0
        agent.search_available = True
        agent.available_buttons = []
        agent.max_turns = config["agent_config"]["max_turns"]

        while True:
            done, role, observation = agent.act(role, observation)
            agent.turn += 1

            if agent.turn > agent.max_turns:
                if agent.shopper_simulator is not None:
                    agent.save_to_json(
                        0, {}, {}, {}, agent.shopper_simulator.messages
                    )
                if shop_env is not None:
                    shop_env.release()
                break

            if done:
                break

    except Exception as e:
        print(f"Error (task {task_id}): {e}")
        traceback.print_exc()
        if shop_env is not None:
            try:
                shop_env.release()
            except Exception:
                pass


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
