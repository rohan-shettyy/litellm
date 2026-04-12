import os
import sys
from unittest.mock import patch
import pytest

# Add the parent directory to the system path to find litellm
sys.path.insert(0, os.path.abspath("."))

from litellm.llms.ollama.completion.transformation import OllamaConfig

class DummyResponse:
    def __init__(self, json_data, status_code=200):
        self._json = json_data
        self.status_code = status_code
    def json(self):
        return self._json

@pytest.mark.parametrize(
    "api_base, expected_url",
    [
        ("http://host:11434/api/chat", "http://host:11434/api/show"),
        ("http://host:11434/api/generate", "http://host:11434/api/show"),
        ("http://host:11434", "http://host:11434/api/show"),
        ("http://host:11434/", "http://host:11434/api/show"),
        ("http://host:11434/api/chat/", "http://host:11434/api/show"),
        ("http://host:11434/api/generate/", "http://host:11434/api/show"),
    ],
)
def test_ollama_get_model_info_url_normalization(api_base, expected_url):
    """
    Tests that get_model_info correctly normalizes the api_base
    before appending /api/show.
    """
    config = OllamaConfig()
    
    with patch("litellm.module_level_client.post") as mock_post:
        mock_post.return_value = DummyResponse({"template": "", "model_info": {}})
        
        config.get_model_info("llama2", api_base=api_base)
        
        # Verify the URL passed to post
        actual_url = mock_post.call_args[1].get("url")
        assert actual_url == expected_url
