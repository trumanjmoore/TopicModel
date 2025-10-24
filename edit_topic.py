from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from keybert import KeyBERT
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
import nltk
from nltk.tokenize import sent_tokenize
from bertopic.backend import Model2VecBackend
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction import text
from sentence_transformers import SentenceTransformer

i = 0
docs = []
with open("abstracts", 'r', encoding='utf-8') as file:
    texts = file.read()
    sentences = sent_tokenize(texts)
    for sentence in sentences:
        if i > 20000:
            break
        docs.append(sentence)
        i += 1

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

topic_model = BERTopic.load("abstracts_model", embedding_model=embedding_model)

topics_to_merge = [[-1, 8], [13, 14], [17, 10]]
#topic_model.merge_topics(docs, topics_to_merge)
print(len(topics_to_merge))
topic_model.reduce_topics(docs, nr_topics=(18-len(topics_to_merge)))

fig = topic_model.visualize_topics()
heat_map = topic_model.visualize_heatmap()
fig.show()
heat_map.show()

hierarchical_topics = topic_model.hierarchical_topics(docs)
tree = topic_model.get_topic_tree(hierarchical_topics)

print(tree)

topic_model.save("abstracts_model", serialization="safetensors", save_ctfidf=True, save_embedding_model=embedding_model)