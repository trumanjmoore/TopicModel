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
from transformers import pipeline

i = 0
docs = []
with open("txt/AAHP 073 Diana Bell 9-5-2009ufdc.txt", 'r', encoding='utf-8') as file:
    texts = file.read()
    sentences = sent_tokenize(texts)
    for sentence in sentences:
        if i > 20000:
            break
        docs.append(sentence)
        i += 1

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

topic_model = BERTopic.load("single_doc_model", embedding_model=embedding_model)

classifier = pipeline("zero-shot-classification", top_n_words=5, model="facebook/bart-large-mnli")

#topics_to_merge = [[-1, 8], [13, 14], [17, 10]]
#topic_model.merge_topics(docs, topics_to_merge)
#print(len(topics_to_merge))
#topic_model.reduce_topics(docs, nr_topics=(18-len(topics_to_merge)))

candidate_labels = ['school', 'military', 'family', 'music', 'religion', 'sports', 'civil rights', 'travel', 'health', 'work']
for i in range(len(topic_model.get_topics())-1):
    sequence_to_classify = " ".join([word for word, _ in topic_model.get_topic(i)[:5]])
    scores = classifier(sequence_to_classify, candidate_labels)
    if scores['scores'][0] > 0.7:
        print(i)
        print(f"{scores['labels'][0]}, {scores['scores'][0]}")
        print(scores['sequence'])

"""fig = topic_model.visualize_topics()
heat_map = topic_model.visualize_heatmap()
fig.show()
heat_map.show()
"""

for t in topic_model.get_topic_info()['Topic']:
    if t == -1:
        continue  # skip outliers
    print(f"Topic {t}:")
    i = 0
    for word, val in topic_model.get_topic(t):
        if i > 5:
            break
        print(f"    {word}")
        i += 1

#topic_model.save("abstracts_model", serialization="safetensors", save_ctfidf=True, save_embedding_model=embedding_model)