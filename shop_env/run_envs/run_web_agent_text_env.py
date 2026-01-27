"""
Test the text gym environment.

TODO: move to testing dir for more rigorous tests
"""
import gym
from rich import print

from web_agent_site.models import LLMPolicy
from web_agent_site.utils import DEBUG_PROD_SIZE

import json
import os
import argparse
from tqdm import tqdm

import traceback

def get_finished_task(out_path):
    json_files = []
    for filename in os.listdir(out_path):
        if filename.endswith(".json"):
            filename = filename.split(".")[0]
            json_files.append(int(filename))
    return json_files

def get_parser():
    parser = argparse.ArgumentParser(description="Run the web agent text environment.")

    # 添加命令行参数
    parser.add_argument("--mode", type=str, choices=["idealab", "whale", "qwen3"], help="")
    parser.add_argument("--model", type=str, default="qwen2.5-72b-instruct", help="")
    parser.add_argument("--out", type=str, default="path", help="")
    parser.add_argument("--eval_num", type=int, default=500, help="")
    parser.add_argument("--split", type=str, default="eval", help="")
    parser.add_argument("--shuffle_goals", type=bool, default=False, help="")
    parser.add_argument("--shuffle_num", type=int, default=20, help="")
    parser.add_argument("--shift_goals", type=bool, default=False, help="")
    parser.add_argument("--if_persona", action='store_true', default=False, help="")
    parser.add_argument("--test_num", type=int, default=None, help="")
    parser.add_argument("--indices_file", type=str, default=None, help="")
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = get_parser()
    MAX_TRY = 30
    model = args.model
    mode = args.mode
    split = args.split
    out_path = f"{args.out}/{model}/{split}"
    shuffle_goals = args.shuffle_goals
    env = gym.make('WebAgentTextEnv-v0', observation_mode='text', num_products=DEBUG_PROD_SIZE, shuffle_goals=shuffle_goals, shuffle_num=args.shuffle_num, shift_goals=args.shift_goals, if_persona=args.if_persona)

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    finished_tasks = get_finished_task(out_path)
    if args.indices_file:
        with open(args.indices_file, "r") as f:
            indices = json.load(f)
        all_tasks = indices
    elif split:
        if "train_persona" in split:
            all_tasks = list(range(1459,4782))
        if 'train' in split:
            all_tasks = list(range(1459,23421))
        elif 'persona' in split:
            all_tasks = list(range(1343))
        elif 'eval' in split:
            all_tasks = list(range(1459))
    else:
        all_tasks = list(range(args.test_num))
    todo_tasks = [i for i in all_tasks if i not in finished_tasks]

    for i in tqdm(todo_tasks, desc="Processing", unit="iter"):
        env.reset(idx=i)
        if env.session in finished_tasks:
            continue
        turn = 0
        done = False
        reward = 0
        status = {}
        policy = LLMPolicy(model, mode)
        observation = env.observation
        try:
            while True:
                if turn >= MAX_TRY:
                    if 'goal' in status:
                        reward_detail, purchase, goal = {}, {}, status['goal']
                    else:
                        reward_detail, purchase, goal = {}, {}, {}
                    result = {"done":done, "turn": turn, "reward": reward, "reward_detail": reward_detail,"purchase": purchase, "goal": goal, "conversation": env.history}
                    break

                if done:
                    reward_detail, purchase, goal = status["reward_detail"], status['purchase'], status['goal']
                    reward_detail_save = {k: v for k, v in reward_detail.items() if not k.startswith("w_")}
                    result = {"done":done, "turn": turn, "reward": reward, "reward_detail": reward_detail_save,"purchase": purchase, "goal": goal, "conversation": env.history}
                    break

                available_actions = env.get_available_actions()
                action = policy.forward(observation, env, available_actions)

                if action is None or '\nAction: ' not in action:
                    observation = "您返回的action无法解析。请检查格式是否正确。"
                else:
                    action = action.split("\nAction: ")[1]
                    observation, status, info = env.step(action)
                    done, reward = status['done'], status['reward']
                turn += 1

            with open(out_path + f"/{env.session}.json","w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()
            continue
    env.close()
