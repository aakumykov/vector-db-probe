import openai
from pathlib import Path
from read_file_from_home import read_file_from_home

YANDEX_MODEL = "yandexgpt-5-lite"
YANDEX_API_KEY = read_file_from_home('.yandex_ai_token')
YANDEX_FOLDER_ID = "b1g2nef5l3idehorhcup"

client = openai.OpenAI(
    api_key=YANDEX_API_KEY,
    base_url="https://ai.api.cloud.yandex.net/v1",
    project= YANDEX_FOLDER_ID
)

response = client.responses.create(
    model=f"gpt://{YANDEX_FOLDER_ID}/{YANDEX_MODEL}",
    input="Придумай 3 необычные идеи для стартапа в сфере путешествий.",
    temperature=0.0,
    max_output_tokens=1500
)

print(response.output[0].content[0].text)