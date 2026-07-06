import os
import yaml
import requests
from pathlib import Path
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import List, Optional

# enviroment
from dotenv import load_dotenv
load_dotenv()  # Loads the variables into the environment


# Configuration
CONFIG_PATH = Path(__file__).with_name("config.yaml")

with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
    CONFIG = yaml.safe_load(config_file) or {}

BASE_URL = CONFIG.get("base_url", "https://integrate.api.adk.com/v1")
APP_NAME = CONFIG.get("app_name", "data_science")
MODEL = CONFIG.get("model_name", "nvidia/nvidia-nemotron-nano-9b-v2")
TEMPERATURE = CONFIG.get("temperature", "0")
MAX_TOKENS = CONFIG.get("max_tokens", "4096")

# Setup HTTP session with retries
def create_session():
    """Create a session with retry logic for better reliability"""
    session = requests.Session()

    # Retry Strategy for Resilient Comunication 
    retry_strategy = Retry(
        total=3, # Try 3 times
        backoff_factor=1, # Wait 1,2,4 seconds between retries
        status_forcelist=[429, 500, 503, 504]
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    return session

class LLMClient:
    """Simple client to talk to AI models."""
    
    def __init__(self):
        # Get API key from environment
        self.api_key = os.getenv("NVIDIA_API_KEY")
        if not self.api_key:
            raise ValueError("Please set NVIDIA_API_KEY environment variable")
        
        self.base_url = BASE_URL.rstrip("/")
        self.model = MODEL
        self.temperature = TEMPERATURE
        self.max_tokens = MAX_TOKENS
        self.session = create_session()
 
    def chat(self, messages: List[dict], tools: Optional[List[dict]] = None) -> dict:
        """Send messages to the AI and get a response."""
        
        # Prepare the request
        request_data = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        
        # Add tools if provided
        if tools:
            request_data["tools"] = tools
            request_data["tool_choice"] = "auto"
        
        # Set up headers
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            # Make the request
            response = self.session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=request_data,
                timeout=(10, 300)  # 10 seconds to connect, 300 seconds to read
            )

            print(f"{self.base_url}/chat/completions")
            print(headers)
            print(request_data)
            
            # Check if request was successful
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to communicate with AI: {e}")

def create_client() -> LLMClient:
    """Create a new LLM client."""
    return LLMClient()
