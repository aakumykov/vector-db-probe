from huggingface_hub import login
from pathlib import Path

def login_to_huggingface(token_path='.hf_token'):
    token = Path(token_path).read_text().strip()
    if not token:
        raise ValueError(f"No or empty token from '{token_path}'")
    login(token)