"""
sentiment_analysis.py

Takes the topics found from the saved BERTopic model and creates a sentiment analysis for each sentence in that topic
Outputs a JSON file directly to an Omeka item through the REST API
"""
import glob
import tkinter as tk
import re
from collections import defaultdict
from nltk.tokenize import sent_tokenize

# Information needed for uploading the file to omeka
BASE_URL = "https://digital.domains.uflib.ufl.edu/omeka/api/media"
KEY_IDENTITY = "0doJ2KWaVenL4xpDvNo4cTtw72pvaPJb"
KEY_CREDENTIAL = "Gr53SxNByDtBbAKQQzfXmZPDPe2mN7au"
FILE_PATH = r"C:\Users\truma\Downloads\treemap_data.json"
FILE_TITLE = "treemap_data.json"
params = {
    "key_identity": KEY_IDENTITY,
    "key_credential": KEY_CREDENTIAL
}

# Load the saved BERTopic model
MODEL_PATH = "TopicModel/full_docs_model.pkl"

# Requirements for topics to pass in order to be considered
# Label score is how closely it relates to one of the predetermined topics
# Classifier words are the most representative words in each topics
# Adding these thresholds makes sure that the topics analyzed are coherent and useful to the project
MIN_LABEL_SCORE = 0.6
MIN_LABEL_SCORE_GAP = 0.10
N_CLASSIFIER_WORDS = 15

# Requirements for sentences in each topic
# Topic probability is how confident BERTopic was that the sentence belongs to that topic
# Semantic similarity is how similar the sentence embedding is to an embedding of a description of the topic
# They are combined with their respective weights to create a "representativeness" that is used to sort sentences
MIN_TOPIC_PROB = 0.50
TOPIC_PROB_WEIGHT = 0.35
MIN_SEMANTIC_SIMILARITY = 0.30
SEMANTIC_WEIGHT = 0.65

# Look through at max the top 20 more representative docs (sentences) for each topic
# Rarely a limit that is reached
MAX_REPR_DOCS = 20

# How fare to grab context
CONTEXT_SENTENCES_BEFORE = 1
CONTEXT_SENTENCES_AFTER = 1

# If a whole sentence (including context) is just a couple lines, dont included it for the visualization
MIN_EXCERPT_WORDS = 4

# Safety net for larger interviews to keep the sentiment analysis model time shorter
SENTIMENT_BATCH_SIZE = 32

# How confident the sentiment model has to be for the sentence to be added
# Very low because I would rather a sentence get added even if the model is unconfident
MIN_SENTIMENT_CONFIDENCE = 0.2

# Clean transcript, remove any names(of interviewers and interviewees) and stop words/artifacts
SPEAKER_PATTERN = re.compile(r"^\s*([A-Za-z]{1,20}):\s*(.*)$")
LEADING_SPEAKER_PATTERN = re.compile(r"^\s*[A-Za-z]{1,20}:\s*")
TEXT_SKIPS = {
    "AAHP", "Samuel", "Joel", "241", "PO", "Gainesville,", "(352)",
    "https://oral.history.ufl.edu", "African", "Abstract:", "Keywords:",
    "For", "Interviewee:", "Transcribed"
}
TRANSCRIPT_ARTIFACT_PATTERN = re.compile(
    r"^\s*(?:\[(?:laughter|inaudible|crosstalk|pause)[^\]]*\]|"
    r"\((?:laughter|inaudible|crosstalk|pause)[^\)]*\))\s*",
    flags=re.IGNORECASE,
)

# Candidate topics, guides the model towards these topics
# Topic descriptions are used for comparing semantic similarity
candidate_labels = [
    "Civil Rights and Discrimination",
    "Education",
    "Health",
    "Religion",
    "Law Enforcement",
    "Military Service",
    "Work",
    "Music",
    "Farming",
    "Books and Reading",
    "Family",
]
TOPIC_DESCRIPTIONS = {
    "Civil Rights and Discrimination": (
        "Experiences of racial discrimination, segregation, civil rights, "
        "racial inequality, protest, integration, or Black political activism."
    ),
    "Education": (
        "Experiences involving schools, teachers, students, classes, education, "
        "colleges, universities, learning, or segregation in education."
    ),
    "Health": (
        "Experiences involving illness, injury, disability, doctors, hospitals, "
        "medical treatment, healthcare, or recovery."
    ),
    "Religion": (
        "Experiences involving churches, worship, prayer, faith, religious "
        "beliefs, clergy, congregations, or religious communities."
    ),
    "Law Enforcement": (
        "Experiences involving police, arrest, law enforcement, courts, jail, "
        "prison, or incarceration."
    ),
    "Military Service": (
        "Experiences involving military service, armed forces, war, combat, "
        "deployment, veterans, military training, or veterans' services."
    ),
    "Work": (
        "Experiences involving jobs, employment, businesses, occupations, "
        "workplaces, wages, promotions, trades, or financial circumstances."
    ),
    "Music": (
        "Experiences involving Music, singing, Musicians, instruments, "
        "performances, bands, or Musical traditions."
    ),
    "Farming": (
        "Experiences involving farms, crops, livestock, agricultural labor, "
        "land cultivation, or rural food production."
    ),
    "Books and Reading": (
        "Experiences involving books, reading, writing, literature, libraries, "
        "publishing, or reading practices."
    ),
    "Family": (
        "Experiences involving parents, children, spouses, marriage, relatives, "
        "household life, family relationships, or the home."
    ),
}


class Window:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Topic And Sentiment Analysis")
        self.window.lift()
        self.window.attributes("-topmost", True)
        self.window.after_idle(self.window.attributes, "-topmost", False)

        main_frame = tk.Frame(self.window)
        main_frame.grid(column=0, row=0, sticky="nswe", padx=10, pady=10)

        frame_top = tk.Frame(main_frame)
        frame_top.grid(column=0, row=0, sticky="nswe")

        frame_bot = tk.Frame(main_frame)
        frame_bot.grid(column=0, row=1, sticky="nswe")

        tk.Label(frame_top, text="Please Enter Omeka Item ID").grid(row=0, column=0)

        self.entry_field = tk.Entry(frame_bot)
        self.entry_field.grid(row=1, column=1, sticky="nswe")

        tk.Button(frame_bot, height=2, width=20, text="Confirm", command=self.take_input).grid(row=1, column=2,
                                                                                               sticky="nswe")

        self.window.bind("<Return>", lambda event: self.take_input())

        self.window.update_idletasks()
        screen_width = self.window.winfo_reqwidth()
        screen_height = self.window.winfo_reqheight()
        x = (self.window.winfo_screenwidth() - screen_width) // 2
        y = (self.window.winfo_screenheight() - screen_height) // 2
        self.window.geometry(f"{screen_width}x{screen_height}+{x}+{y}")
        self.window.resizable(False, False)

        self.item_id = None

    def take_input(self):
        value = self.entry_field.get().strip()
        if value:
            self.item_id = value
            self.window.destroy()

    def on_closing(self):
        self.item_id = None
        self.window.destroy()

    def wait_for_input(self):
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.mainloop()
        return self.item_id


# Clean and normalize sentences for comparison and the visualization
def clean_display_sentence(sentence):
    sentence = LEADING_SPEAKER_PATTERN.sub("", sentence).strip()
    sentence = TRANSCRIPT_ARTIFACT_PATTERN.sub("", sentence).strip()
    sentence = re.sub(r"\s+", " ", sentence)
    return sentence


def normalized_text(text):
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def is_good_excerpt(text):
    clean = clean_display_sentence(text)
    words = re.findall(r"\b[\w’'-]+\b", clean)

    if MIN_EXCERPT_WORDS >= len(words):
        return False
    if len(re.findall(r"[A-Za-z]", clean)) < 20:
        return False
    if clean.endswith("?"):
        return False

    return True


# Get the probability for each topic that BERTopic found
def assigned_probability(probabilities, index):
    if probabilities is None:
        return 1.0
    return float(probabilities[index])


# Get the context around a representative setnence
# This gives the user a little more information when looking at the visualization
def build_topic_contexts(sentences, turn_ids):
    contexts = []
    for index, sentence in enumerate(sentences):
        turn_id = turn_ids[index]
        start = max(0, index - CONTEXT_SENTENCES_BEFORE)
        end = min(len(sentences), index + CONTEXT_SENTENCES_AFTER + 1)
        nearby = [sentences[position] for position in range(start, end) if turn_ids[position] == turn_id]

        context = " ".join(nearby) if nearby else sentence
        contexts.append(context)
    return contexts


# Load the transcript, seperate speakers because we dont care about the interviewer
def load_transcript():
    turns = []
    current_speaker = None
    turn_id = -1
    filelist = glob.glob("Input\\*.txt")
    for file in filelist:
        with open(file, "r", encoding="utf-8") as source:
            for raw_line in source:
                line = raw_line.rstrip("\n")
                stripped = line.strip()
                if not stripped:
                    continue

                first_word = stripped.split()[0]
                if first_word in TEXT_SKIPS:
                    continue

                speaker_match = SPEAKER_PATTERN.match(line)
                if speaker_match:
                    current_speaker = speaker_match.group(1).upper()
                    line = speaker_match.group(2).strip()
                    turn_id += 1
                elif current_speaker is None:
                    continue

                if not line.strip():
                    continue

                turns.append((current_speaker, line, turn_id))

        if not turns:
            return [], []

        word_counts = defaultdict(int)
        for speaker, text, _ in turns:
            word_counts[speaker] += len(text.split())

        narrator = max(word_counts, key=word_counts.get)
        print(f"Detected narrator: {narrator}")

        docs = []
        turn_ids = []
        for speaker, text, tid in turns:
            if speaker != narrator:
                continue
            for sentence in sent_tokenize(text):
                sentence = clean_display_sentence(sentence)
                if sentence:
                    docs.append(sentence)
                    turn_ids.append(tid)

    return docs, turn_ids


# If there are any sentences that share context in the same topic, combine them into one entry in the visualization
def merge_overlapping_excerpts(excerpts, all_docs):
    position = {}
    for idx, sentence in enumerate(all_docs):
        if sentence not in position:
            position[sentence] = idx

    excerpts = [dict(e) for e in excerpts]

    i = 0
    while i < len(excerpts):
        j = i + 1
        while j < len(excerpts):
            sentence_i = sent_tokenize(excerpts[i]["name"])
            sentence_j = sent_tokenize(excerpts[j]["name"])

            shared = set(sentence_i) & set(sentence_j)

            if shared:
                seen = set()
                all_sentences = []
                for s in sentence_i + sentence_j:
                    if s not in seen:
                        seen.add(s)
                        all_sentences.append(s)

                all_sentences.sort(key=lambda s: position.get(s, float('inf')))

                base = excerpts[i] if (excerpts[i]["prob"] >= excerpts[j]["prob"]) else excerpts[j]

                merged = dict(base)
                merged["name"] = " ".join(all_sentences)
                excerpts[i] = merged
                excerpts.pop(j)
            else:
                j += 1

        i += 1

    return excerpts


def score_sentiment(texts, sentiment_pipeline):
    results = []
    for start in range(0, len(texts), SENTIMENT_BATCH_SIZE):
        batch = texts[start:start + SENTIMENT_BATCH_SIZE]
        raw_results = sentiment_pipeline(batch)

        for scores in raw_results:
            by_label = {item["label"].lower(): item["score"] for item in scores}
            positive = (by_label.get("positive", 0.0) + by_label.get("very positive", 0.0))
            negative = (by_label.get("negative", 0.0) + by_label.get("very negative", 0.0))
            neutral = by_label.get("neutral", 0.0)
            signed_score = positive - negative

            if positive > negative and positive > neutral:
                label = "POSITIVE"
            elif negative > positive and negative > neutral:
                label = "NEGATIVE"
            else:
                label = "NEUTRAL"

            results.append((label, signed_score))

    return results

def main(item_id):
    import json
    import requests
    import numpy as np
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from transformers import pipeline

    docs, turn_ids = load_transcript()

    # Contexts for each sentence, it is these sentences that will be fed into the topic and embedding models
    topic_contexts = build_topic_contexts(docs, turn_ids)

    # Load the embedding model and the saved BERTopic topic model
    embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    topic_model = BERTopic.load(MODEL_PATH, embedding_model=embedding_model)

    # Run the embedding model on the sentences (with context)
    context_embeddings = embedding_model.encode(topic_contexts, show_progress_bar=True, convert_to_numpy=True,
                                                normalize_embeddings=True, )
    # Run the topic model on the sentences (with context)
    topics_assigned, probs = topic_model.transform(topic_contexts, embeddings=context_embeddings)

    # Remove the outlier topic
    unique_topics = sorted({int(topic) for topic in topics_assigned if int(topic) != -1})
    # Cache the topics to prevent repeated topic_model_get_topic calls
    topic_words_cache = {}
    for topic_id in unique_topics:
        topic_words = topic_model.get_topic(topic_id)
        if topic_words:
            topic_words_cache[topic_id] = topic_words

    print(f"Found {len(topic_words_cache)} non-outlier BERTopic topics")

    candidates_by_label = defaultdict(list)
    # Load the classifier model
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    # Load an embedding model for semantic comparison of topics and topic descriptions
    label_embeddings = {label: embedding_model.encode(description, convert_to_numpy=True, normalize_embeddings=True)
                        for label, description in TOPIC_DESCRIPTIONS.items()}

    # Classify topics based on candidate topics and combine topics that belong the same candidate topic
    for topic_id, topic_words_with_scores in topic_words_cache.items():
        topic_words = [word for word, _ in topic_words_with_scores[:N_CLASSIFIER_WORDS]]
        keyword_sequence = " ".join(topic_words)
        result = classifier(keyword_sequence, candidate_labels)

        top_label = result["labels"][0]
        top_score = float(result["scores"][0])
        second_label = result["labels"][1]
        second_score = float(result["scores"][1])
        score_gap = top_score - second_score

        if top_score < MIN_LABEL_SCORE or score_gap < MIN_LABEL_SCORE_GAP:
            continue

        topic_embedding = label_embeddings[top_label]

        for index, assigned_topic in enumerate(topics_assigned):
            if int(assigned_topic) != topic_id:
                continue

            context_passage = topic_contexts[index]
            if not is_good_excerpt(context_passage):
                continue

            topic_probability = assigned_probability(probs, index)
            if topic_probability < MIN_TOPIC_PROB:
                continue

            semantic_similarity = float(np.dot(context_embeddings[index], topic_embedding))
            if semantic_similarity < MIN_SEMANTIC_SIMILARITY:
                continue

            representativeness = (TOPIC_PROB_WEIGHT * topic_probability + SEMANTIC_WEIGHT * semantic_similarity)

            candidates_by_label[top_label].append({
                "context": context_passage,
                "representativeness": representativeness,
            })

    # Deduplicate and normalize sentences in a topic and rank them by representativeness
    selected_by_label = {}
    for label, candidates in candidates_by_label.items():
        best_by_text = {}
        for candidate in candidates:
            key = normalized_text(candidate["context"])
            previous = best_by_text.get(key)
            if previous is None or (candidate["representativeness"] > previous["representativeness"]):
                best_by_text[key] = candidate

        ranked = sorted(best_by_text.values(), key=lambda item: item["representativeness"], reverse=True)
        selected_by_label[label] = ranked[:MAX_REPR_DOCS]

    # Build the sentiment model
    sentiment_pipeline = pipeline("sentiment-analysis", model="tabularisai/multilingual-sentiment-analysis",
                                  top_k=None, truncation=True, max_length=512)

    # Run the sentiment model on all the sentences that build a topic
    children = []
    for topic_label, selected in selected_by_label.items():
        if not selected:
            continue
        display_sentiments = score_sentiment([item["context"] for item in selected], sentiment_pipeline)

        confident_sentiments = [result for result in display_sentiments]
        if confident_sentiments:
            signed_scores = [signed for _, signed in confident_sentiments]
            topic_mean = float(np.mean(signed_scores))
        else:
            topic_mean = 0.0

        label_counts = defaultdict(int)
        for sentiment_label, _ in display_sentiments:
            label_counts[sentiment_label] += 1

        # Assign the topic sentiment as the mode sentiment of its sentences
        # If there is a tie, then use the mean sentiment score to settle the tie
        confident_counts = {label: label_counts[label] for label in ("POSITIVE", "NEUTRAL", "NEGATIVE")}
        largest_count = max(confident_counts.values())
        tied_labels = [label for label, count in confident_counts.items() if count == largest_count]
        if len(tied_labels) == 1:
            topic_sentiment = tied_labels[0]
        elif topic_mean > 0.10 and "POSITIVE" in tied_labels:
            topic_sentiment = "POSITIVE"
        elif topic_mean < -0.10 and "NEGATIVE" in tied_labels:
            topic_sentiment = "NEGATIVE"
        else:
            topic_sentiment = "NEUTRAL"

        # Create the ds for the json file
        sentence_children = []
        for item, display_result in zip(selected, display_sentiments):
            sentiment_label, signed_score = display_result
            sentence_children.append({"name": item["context"], "prob": round(float(item["representativeness"]), 4),
                                      "sentiment": sentiment_label})

        # Because there may be a lot of overlapping sentences, merge them into one entry for the visualization
        sentence_children = merge_overlapping_excerpts(sentence_children, docs)

        children.append({"name": topic_label, "sentiment": topic_sentiment, "children": sentence_children})

    # Show topics in order of how many sentences
    children.sort(key=lambda topic: len(topic["children"]), reverse=True)

    json_dict = {
        "name": "Interview Topics",
        "children": children
    }

    # Final schema guard: stop rather than writing a malformed file if a complete
    # Python list is accidentally assigned to either public text field.
    for topic in json_dict["children"]:
        for excerpt in topic.get("children", []):
            if not isinstance(excerpt.get("name"), str):
                raise TypeError(
                    f"Excerpt name under '{topic.get('name')}' must be a string."
                )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as destination:
        json.dump(json_dict, destination, indent=2, ensure_ascii=False)

    print(f"\nSaved to {OUTPUT_FILE}")
    print(
        f"Topics: {len(children)}, Total excerpts: "
        f"{sum(len(topic['children']) for topic in children)}"
    )

    metadata = {
        "o:ingester": "upload",
        "file_index": 0,
        "o:item": {"o:id": item_id},
        "dcterms:title": [{"type": "literal", "property_id": 1, "@value": FILE_TITLE}]
    }
    headers = {"User-Agent": "curl/8.4.0"}
    with open(FILE_PATH, "rb") as f:
        files = {"file[0]": (FILE_TITLE, f, "application/json")}
        data = {"data": json.dumps(metadata)}
        response = requests.post(BASE_URL, params=params, files=files, data=data, headers=headers)
    result = response.json()
    if "o:id" in result:
        print(f"✅ Upload successful!")
        print(f"   Media ID : {result['o:id']}")
        print(f"   Title    : {result['o:title']}")
        print(f"   File URL : {result['o:original_url']}")
    else:
        print(f"❌ Upload failed: {result}")


if __name__ == "__main__":
    item_id = Window().wait_for_input()
    print(f"Got item_id: {item_id!r}")
    if item_id is None:
        print("Cancelled.")
        raise SystemExit(0)

    main(item_id)