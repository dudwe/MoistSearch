from venv import create
from networkx import in_degree_centrality
from sentence_transformers import SentenceTransformer
import csv
import numpy as np
import faiss
from torch import embedding_renorm_

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loaded model!")

data = []
embeddings = []


def create_searchable_text(row):
    text = ""
    for k,v in row.items():
        if isinstance(v,list):
            text += f"{k}: {' '.join(map(str,v))} "
        else:
            text += f"{k}: {v} "
    return text.strip()

with open('moistmeter.csv', newline='') as csvfile:
    spamreader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
    for row in spamreader:
        row["Genre"] = [x.strip() for x in row["Genre"].split(",")]  
        searchtext = create_searchable_text(row)
        print(row)
        data.append(row)
        embedding = model.encode(searchtext)
        embeddings.append(embedding)

embeddings = np.array(embeddings).astype('float32')

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

def search(query,k=5):
    query_vector = model.encode([query])
    distances, indices = index.search(query_vector, k)

    results = []
    for i in indices[0]:
        print(i)
        results.append(data[i])
    print(distances,indices)
    return results
results = search("find me the best monkey movie")
print(results)
    