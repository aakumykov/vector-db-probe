#!.venv/bin/python

from pathlib import Path
from huggingface_hub import login
from sentence_transformers import SentenceTransformer


def auth_in_hf(token_path='.hf_token'):
    token = Path(token_path).read_text().strip()
    if not token:
        raise ValueError(f"No or empty token from '{token_path}'")
    login(token)


auth_in_hf()

inputs = [
    # 
    "paraphrase: В Ярославской области разрешили работу бань, но без посетителей",
    "categorize_entailment: Женщину доставили в больницу, за ее жизнь сейчас борются врачи.",
    "search_query: Сколько программистов нужно, чтобы вкрутить лампочку?",
    # 
    "paraphrase: Ярославским баням разрешили работать без посетителей",
    "categorize_entailment: Женщину спасают врачи.",
    "search_document: Чтобы вкрутить лампочку, требуется три программиста: один напишет программу извлечения лампочки, другой — вкручивания лампочки, а третий проведет тестирование."
]

# loads model with CLS pooling
model = SentenceTransformer("ai-forever/FRIDA")

# embeddings are normalized by default
embeddings = model.encode(inputs, convert_to_tensor=True)

sim_scores = embeddings[:3] @ embeddings[3:].T
print(sim_scores.diag().tolist())
# [0.9360026717185974, 0.8591331243515015, 0.7285830974578857]
