import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from bertopic import BERTopic
from nltk.tokenize import sent_tokenize
from bertopic.vectorizers import ClassTfidfTransformer

candidate_seed_topics = [
        ["civil rights", "segregated", "rights", "freedom", "justice", "activist", "protest", "equality", "movement",
         "struggle", "rights", "advocacy", "boycott", "black history", "freedom", "race", "empowerment", "segregation",
         "racism", "discrimination", "jim crow", "inequality", "prejudice", "desegregation", "rights", "black culture",
         "oppression", "integration", "excluded", "black", "african american", "black students", "black figures",
         "predominantly white", "inferior", "separate"],

        ["education", "school", "teacher", "student", "college", "university", "campus", "classroom", "curriculum",
         "graduate", "degree", "professor", "scholarship", "academic"],

        ["job", "work", "employment", "money", "income", "occupation", "labor", "factory", "union", "industry"],

        ["farm", "crops", "harvest", "soil", "plant", "livestock", "field", "farmer", "orchard", "tractor", "barn"],

        ["family", "children", "parents", "home", "siblings", "relatives", "mother", "father", "grandparents"],

        ["government", "law", "election", "vote", "democracy", "congress", "senate", "representation", "justice",
         "legislation"],

        ["health", "doctor", "hospital", "medicine", "nurse", "illness", "sick", "disease", "treatment", "patient",
         "clinic", "wellness", "therapy", "surgery", "healthcare", "cure"],

        ["slavery", "enslaved", "slave", "plantation", "abolition", "bondage", "oppression", "servitude", "cotton",
         "emancipation"],

        ["religion", "church", "faith", "worship", "spiritual", "bible", "preacher", "pastor", "god", "prayer"],

        ["military", "army", "service", "war", "veteran", "battle", "soldier", "duty", "deployment", "combat"],

        ["music", "song", "art", "painting", "artist", "dance", "instrument", "performance", "singing", "culture",
         "poetry", "blues"],

        ["sports", "game", "team", "football", "basketball", "baseball", "athlete", "competition", "score", "boxing",
         "coach", "league", "championship", "track"],

        ["police", "cop", "law enforcement", "arrest", "crime", "court", "sheriff", "law", "jail", "prison"]
    ]

text_skips = ["AAHP", "Samuel", "Joel", "241", "PO", "Gainesville,", "(352)", "https://oral.history.ufl.edu", "African",
              "Abstract:", "Keywords:", "For", "Interviewee:", "Transcribed"]

# load all the interview transcripts
docs = []
with open("txt/AAHP 073 Diana Bell 9-5-2009ufdc.txt", 'r', encoding='utf-8') as file:
    texts = file.read()
    for part in texts.split("\n"):
        if part.split() and part.split()[0] in text_skips:
            continue
        sentences = sent_tokenize(part)
        for sentence in sentences:
            docs.append(sentence)

ctfidf_model = ClassTfidfTransformer(
    seed_words=candidate_seed_topics,
    seed_multiplier=500
)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
topic_model = BERTopic.load("single_doc_model", embedding_model=embedding_model)

classifier = pipeline("zero-shot-classification", top_n_words=5, model="facebook/bart-large-mnli")

model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model=model_name,
    tokenizer=model_name
)

doc_topic = pd.DataFrame({
  'Topic':topic_model.topics_,
  'ID':range(len(topic_model.topics_)),
  'Document':docs}
) # topics and docs combined, required by internal functions
#topic_model._save_representative_docs(doc_topic)
repr_docs, _, _, _=  topic_model._extract_representative_docs(
    topic_model.c_tf_idf_,
    doc_topic,
    topic_model.topic_representations_,
    nr_samples=1000,
    nr_repr_docs=5
)
topic_model.representative_docs_ = repr_docs

sentiments = {}
candidate_labels = ['school', 'military', 'family', 'music', 'religion', 'sports', 'civil rights', 'travel', 'health', 'work']
for i in range(len(topic_model.get_topics())-1):
    sequence_to_classify = " ".join([word for word, _ in topic_model.get_topic(i)[:5]])
    scores = classifier(sequence_to_classify, candidate_labels)
    if scores['scores'][0] > 0.7:
        print(i)
        print(f"{scores['labels'][0]}, {scores['labels'][1]}, {scores['scores'][0]}, {scores['scores'][1]}")
        print(scores['sequence'])
        all_sentences = " ".join([doc.strip("\n") for doc in topic_model.get_representative_docs(i)])
        print(all_sentences)
        sentiments[i] = sentiment_pipeline(all_sentences)[0]
        print(sentiments[i])

