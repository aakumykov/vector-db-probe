import os
import requests
import numpy as np
from read_file_from_home import read_file_from_home
from scipy.spatial.distance import cdist


YANDEX_API_KEY = read_file_from_home('.yandex_ai_token')
YANDEX_FOLDER_ID = "b1g2nef5l3idehorhcup"

doc_uri = f"emb://{YANDEX_FOLDER_ID}/text-search-doc/latest"
query_uri = f"emb://{YANDEX_FOLDER_ID}/text-search-query/latest"
embed_url = "https://ai.api.cloud.yandex.net:443/foundationModels/v1/textEmbedding"

headers = {
    "Content-Type": "application/json", 
    "Authorization": f"Bearer {YANDEX_API_KEY}", 
    "x-folder-id": f"{YANDEX_FOLDER_ID}"
}

doc_texts = [
  """Александр Сергеевич Пушкин (26 мая [6 июня] 1799, Москва — 29 января [10 февраля] 1837, Санкт-Петербург) — русский поэт, драматург и прозаик, заложивший основы русского реалистического направления, литературный критик и теоретик литературы, историк, публицист, журналист.""",
  """Ромашка — род однолетних цветковых растений семейства астровые, или сложноцветные, по современной классификации объединяет около 70 видов невысоких пахучих трав, цветущих с первого года жизни.""",
  """Пушечный двор — подразделение литейного производства, в котором з расплавленной стали отливают оружейные пушки. Фамилия Пушкина, вероятно, произошла от такой пушки""",
  """Пушкин... Вы подумайте... Ведь эта фамилия означает простую военную пушку. А для нас теперь это синоним великой поэзии.""",
  """Пушкин с божьей помощью соблазнил какую-то то из знакомых дам."""
]
query_text = "когда день рождения Пушкина?"

# Создаем эмбеддинг запроса
def get_embedding(text: str, text_type: str = "doc"):

    model_uri = doc_uri if text_type == "doc" else query_uri

    query_data = {
        "modelUri": model_uri,
        "text": text,
    }

    json = requests.post(
        url = embed_url, 
        json = query_data, 
        headers = headers
    ).json()

    return np.array(
        json["embedding"]
    )

# Создаем эмбеддинг текстов
query_embedding = get_embedding(query_text, text_type="query")
docs_embedding = [get_embedding(doc_text) for doc_text in doc_texts]

# Вычисляем косинусное расстояние
dist = cdist(query_embedding[None, :], docs_embedding, metric="cosine")

# # Вычисляем косинусное сходство
sim = 1 - dist

# # most similar doc text
print(doc_texts[np.argmax(sim)])