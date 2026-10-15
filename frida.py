#!.venv/bin/python

import chromadb
from chromadb.types import Collection
from sentence_transformers import SentenceTransformer
from for_text_in_files import for_text_in_files
from login_to_huggingface import login_to_huggingface


login_to_huggingface()


chromadb_client = chromadb.Client()
chromadb_collection = chromadb_client.create_collection(name="frida_test_1")

frida_model = SentenceTransformer("ai-forever/FRIDA")


def process_text_to_embedding(file_name: str, text: str, model, db_collection: Collection):
    e = text2embedding(text, model)
    store_embedding_in_database(
        collection = db_collection,
        emb = e,
        emb_id = file_name,
        emb_metadata = { file_name: file_name},
        orig_text = text
    )


def text2embedding(t: str, model: SentenceTransformer):
    # TODO: попробовать convert_to_tensor=True
    return  model.encode(t, convert_to_numpy=True)


def store_embedding_in_database(
        collection: Collection,
        emb,
        emb_id: str,
        emb_metadata: dict,
        orig_text: str,
):
    collection.add(
        embeddings=[emb],
        ids=[emb_id],
        documents=[orig_text],
        metadatas = [emb_metadata]
    )


for_text_in_files("txt", process_text_to_embedding)

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
