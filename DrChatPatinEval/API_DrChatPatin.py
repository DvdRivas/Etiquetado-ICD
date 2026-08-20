# IMPORTS 
import fastapi_poe as fp
import nest_asyncio
from fastapi import FastAPI
import os 
import ast 
import torch 

# ============================================================================================================== #
# CREACION DEL CORPUS APARTIR DE LA BASE DE DATOS DE NORD
# def create_corpus(): 
#     corpus = []
#     for data in os.listdir('NORD_DB'):
#         rutes = os.path.join('NORD_DB', data)
    
#         with open(rutes, 'r', encoding='utf-8') as doc:
#             corpus.append(doc.read())
    
#     with open('corpus.txt', 'w', encoding='utf-8') as doc:    
#         doc.write("corpus = [\n")
#         for data in corpus:
#             doc.write(f" {repr(data)}, \n")
#         doc.write("]")
#     print(corpus)
    
# create_corpus()

# ============================================================================================================== #
# CARGA DEL CORPUS
def load_corpus() -> list:
    with open('corpus.txt', 'r', encoding='utf8') as file:
        contenido_completo = file.read()
            
            # Encuentra el inicio de la lista (después de 'corpus = [')
        inicio_lista = contenido_completo.find('[')
        
        # Extrae solo la parte de la lista '[...]'
        string_de_la_lista = contenido_completo[inicio_lista:]
        
        # Convierte de forma segura el string a una lista de Python
        corpus = ast.literal_eval(string_de_la_lista)
        
    print(len(corpus))
    return corpus

corpus = load_corpus()
# ============================================================================================================== #
# from transformers import AutoModel
# # Initialize the model
# model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)

# texts = [
#     "Follow the white rabbit.",  # English
#     "Sigue al conejo blanco.",  # Spanish
#     "Suis le lapin blanc.",  # French
#     "跟着白兔走。",  # Chinese
#     "اتبع الأرنب الأبيض.",  # Arabic
#     "Folge dem weißen Kaninchen.",  # German
# ]

# # When calling the `encode` function, you can choose a `task` based on the use case:
# # 'retrieval.query', 'retrieval.passage', 'separation', 'classification', 'text-matching'
# # Alternatively, you can choose not to pass a `task`, and no specific LoRA adapter will be used.
# embeddings = model.encode(texts, task="text-matching")

# # Compute similarities
# print(embeddings[0] @ embeddings[1].T)
# ============================================================================================================== #

# Requires transformers>=4.51.0
# Requires sentence-transformers>=2.7.0

from sentence_transformers import SentenceTransformer

# Load the model
model = SentenceTransformer("Qwen/Qwen3-Embedding-4B")

corpus_embeddings = model.encode_document(corpus)
query_embbeding = model.encode_query('Can you provide information about Acrodysostosis?')

similarity_scores = model.similarity(query_embbeding, corpus_embeddings)[0]
scores, indices = torch.topk(similarity_scores, k = 5)

for score, idx in zip(scores, indices):
    print(f"(Score: {score:.4f})", corpus[idx])
# ============================================================================================================== #
# # KEYs DECLARATIONS
# POE_KEY = "wkyPK_lCauPp0psjBeyZWWwqT69_mie1Q4zOaAgA0rQ"

# # READ THE QUERY
# with open ("query.txt", "r", encoding="utf-8") as data:
#     query = data.read()

# # API POE USE
# def UseAPI(query: string, ins: string) -> string: 
#     nest_asyncio.apply()
#     message = fp.ProtocolMessage(role="user", content=f"{ins} \n {query}")
#     ans = ""

#     for partial in fp.get_bot_response_sync(messages=[message], bot_name="DrChatPatin-20B", api_key=POE_KEY):
#         ans += partial.text

#     return(ans)

# # CREATION OF OUR OWN API 
# app = FastAPI(
#     title = 'DrChatPatin API Service',
#     description='abc',
#     version='0.1.0',
    
# )

# @app.get('/DifferentialDiagnosis', tags=['Differetial Diangonsis'])
# def df(query:string):
#     ins = ""
#     return UseAPI(query, ins)

# @app.get('/DifferentialDiagnosis2ndT', tags=['Differetial Diangonsis'])
# def df(query:string):
#     ins = ""
#     ans = UseAPI(query, ins)
#     ins = "Process the following information about a diagnosis, analize the data again. Do a second thougth considering the following the most relevant information, and then, provide a diferential diagnosis:"
#     return UseAPI(ans, ins)



# COMMANDS
# RUN THE CODE 
# & C:/Users/sweet/anaconda3/python.exe c:/Users/sweet/Python/API_DrChatPatin.py
# RUN THE API SERVICE
# fastapi dev API_DrChatPatin.py
