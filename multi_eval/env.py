"""ShopEnv module: implements main logic for shopping environment interface."""

import logging
from typing import Any, Dict, Optional

import requests
import pdb
logger = logging.getLogger(__name__)

API_PATH = "/api/shop_agent"

REQUEST_TIMEOUT = 30


class ShopEnv:
    """Shopping environment class for interacting with shopping environment API."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize shopping environment.

        Args:
            config: Configuration dictionary containing:
                - base_url: Base URL for API service (required)
                - if_persona: Whether to enable persona mode (optional, default False)

        Raises:
            KeyError: When required 'base_url' key is missing from config
        """
        if "base_url" not in config:
            raise KeyError("Missing required 'base_url' key in config")

        self.base_url = f"{config['base_url']}{API_PATH}"
        self.if_persona = config.get("if_persona", False)
        self.task_id: Optional[str] = None
        self.env_idx: Optional[int] = None

        logger.info(f"ShopEnv initialized, base_url: {self.base_url}, if_persona: {self.if_persona}")

    def _make_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send POST request to API server.

        Args:
            data: Request data dictionary

        Returns:
            Dict[str, Any]: Content of 'result' field from API response

        Raises:
            requests.RequestException: When HTTP request fails
            KeyError: When 'result' field is missing from response
            ValueError: When response status code is not 200
        """
        try:
            response = requests.post(
                self.base_url, json=data, timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()

            response_data = response.json()
            if "result" not in response_data:
                raise KeyError("Missing 'result' field in API response")

            return response_data["result"]

        except requests.exceptions.Timeout:
            logger.error(f"Request timeout: {self.base_url}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP request failed: {e}")
            raise
        except ValueError as e:
            logger.error(f"Response parsing failed: {e}")
            raise

    def reset(self, task_id: str) -> Dict[str, Any]:
        """
        Initialize a new shopping environment.

        Args:
            task_id: Task ID

        Returns:
            Dict[str, Any]: Reset environment information, containing instruction, env_idx, etc.

        Raises:
            requests.RequestException: When API request fails
            KeyError: When 'env_idx' field is missing from response
        """
        self.task_id = task_id
        data = {"action": "reset", "idx": task_id}

        logger.info(f"Resetting environment, task_id: {task_id}")

        result = self._make_request(data)
        
        if "env_idx" not in result:
            raise KeyError("Missing 'env_idx' field in API response")

        if self.if_persona:
            result["instruction"] = result["instruction_simple"]
            del result["instruction_simple"]

        self.env_idx = result["env_idx"]
        logger.info(f"Environment reset successful, env_idx: {self.env_idx}")

        return result

    def release(self) -> Dict[str, Any]:
        """
        Release shopping environment resources.

        Returns:
            Dict[str, Any]: Result of release operation

        Raises:
            ValueError: When environment is not initialized (env_idx is None)
            requests.RequestException: When API request fails
        """
        if self.env_idx is None:
            raise ValueError("Environment not initialized, please call reset() first")

        data = {"action": "release_one", "env_idx": self.env_idx}

        logger.info(f"Releasing environment resources, env_idx: {self.env_idx}")

        result = self._make_request(data)

        self.env_idx = None
        self.task_id = None

        logger.info("Environment resources released successfully")

        return result

    def interact(self, action_text: str) -> Dict[str, Any]:
        """
        Interact with environment.

        Args:
            action_text: Generated Action string, e.g.:
                "search['find a kite light, 56-head flash edge light, budget within 8000.']"
                or "click[product name]"

        Returns:
            Dict[str, Any]: Environment state information after interaction

        Raises:
            ValueError: When environment is not initialized (env_idx is None)
            requests.RequestException: When API request fails
        """
        if self.env_idx is None:
            raise ValueError("Environment not initialized, please call reset() to initialize")

        data = {
            "action": "interact",
            "env_idx": self.env_idx,
            "response": action_text,
        }

        logger.debug(f"Interacting with environment, action_text: {action_text[:100]}...")

        result = self._make_request(data)

        return result