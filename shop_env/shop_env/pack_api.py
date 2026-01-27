import sys
import logging
import time
from typing import Dict, Any, Optional, Set, List

import gym
from flask import Flask, request, jsonify, Response

sys.path.append("../")
from shop_agent import shop_agent
from web_agent_site.utils import DEBUG_PROD_SIZE
from web_agent_site.envs import WebAgentSiteEnv

# Constants
LOG_FILE = "shop_agent.log"
MAX_RETRIES = 5
RETRY_DELAY_SECONDS = 5
DEFAULT_ENV_MAX_NUM = 20
SERVER_HOST = '0.0.0.0'
SERVER_PORT = 5000

# Global variables
envs: List[Any] = []
free_env_index: Set[int] = set()
env_max_num: int = DEFAULT_ENV_MAX_NUM

# Configure logging format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)


@app.route('/api/shop_agent', methods=['POST'])
def api_some_function() -> Response:
    """
    API endpoint for shop agent operations.
    
    Handles three types of actions:
    - release_all: Release all environments
    - release_one: Release a specific environment
    - reset/interact: Process shop agent actions
    
    Returns:
        JSON response with result or error message
    """
    data = request.json
    if data is None:
        logger.error("[Error] No JSON data provided in request")
        return jsonify({'result': {"error": "No JSON data provided"}})
    
    action = data.get('action')
    env_idx = data.get('env_idx', None)
    response = data.get('response', None)
    idx = data.get('idx', None)
    try:
        # Release all environments
        if action == 'release_all':
            for i in range(env_max_num):
                if i not in free_env_index:
                    free_env_index.add(i)
            logger.info("[Init] All environments have been initialized")
            return jsonify({'result': {"message": "All environments have been initialized"}})

        # Release one environment
        if action == 'release_one':
            if env_idx is not None and isinstance(env_idx, int):
                if env_idx not in free_env_index:
                    free_env_index.add(env_idx)
                    logger.info(f"[Release] Environment {env_idx} has been released")
                    return jsonify({'result': {"message": f"Environment {env_idx} has been released"}})
                else:
                    logger.warning(f"[Release] Environment {env_idx} is already free, no need to release again")
                    return jsonify({'result': {"message": f"Environment {env_idx} is already free"}})
            else:
                logger.error("[Error] No valid environment index provided")
                return jsonify({'result': {"error": "No valid environment index provided"}})

        # If env_idx is not provided, assign an available env_idx
        if env_idx is None:
            retry_count = 0
            while retry_count < MAX_RETRIES:
                if len(free_env_index) > 0:
                    env_idx = free_env_index.pop()
                    break
                retry_count += 1
                logger.info(f"[Retry {retry_count}/{MAX_RETRIES}] No available environment index, retrying in {RETRY_DELAY_SECONDS} seconds...")
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                # Reached max retries but still couldn't get env_idx
                logger.error("[Error] Reached max retries, unable to get available environment")
                return jsonify({'result': {'error': 'Unable to get available environment resource, please try again later'}})

        # Call shop_agent function
        result = shop_agent(envs[env_idx], env_idx, action, idx, response)

        # If task is over, release the environment
        if 'over' in result and result['over']:
            free_env_index.add(env_idx)
            logger.info(f"[Task Over] Environment {env_idx} has been released")

    except Exception as e:
        logger.exception(f"[Exception] Exception occurred while processing request: {str(e)}")
        return jsonify({'result': {'error': str(e)}})

    return jsonify({'result': result})


def initialize_environments() -> None:
    """
    Initialize all environments and add them to the free environment index.
    """
    global envs, free_env_index, env_max_num
    
    envs = []
    free_env_index = set()
    
    for i in range(env_max_num):
        logger.info(f"Environment {i} is being initialized")
        free_env_index.add(i)
        env = gym.make(
            'WebAgentTextEnv-v0',
            observation_mode='text',
            split="train",
            num_products=DEBUG_PROD_SIZE
        )
        envs.append(env)


if __name__ == '__main__':
    initialize_environments()
    app.run(host=SERVER_HOST, port=SERVER_PORT)
