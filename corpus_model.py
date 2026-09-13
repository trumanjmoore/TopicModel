"""
corpus_model.py

Trains a BERTopic model on all(currently available as of 09/2026) interview transcripts
Saves the BERTopic model (with pickle) that the sentiment-analysis pipeline can load directly
"""

import re
import sys
from pathlib import Path
import torch
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

# CUDA: Speeds up transformation by pipelining on gpu
# NOTE: ONLY WORKS ON NVIDIA GPU, can run on cpu instead which is slower
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIPELINE_DEVICE = 0 if DEVICE == "cuda" else -1
TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
if DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else:
    print("CUDA is not available for your GPU. Training is slower on CPU.")

# Whereever Transcripts are held
TRANSCRIPT_DIR = Path("txt")
# Where the model should be saved after fitting
MODEL_PATH = Path("TopicModel/full_docs_model.pkl")

# How many sentences you want to take from the transcripts
# There was about ~330000 at the time of fitting, it was only fit on half because of how long it takes
MAX_SENTENCES = 150000
# If you are using cuda you cant do more batches (faster work)
EMBEDDING_BATCH = 128 if DEVICE == "cuda" else 32
ZERO_SHOT_BATCH = 16 if DEVICE == "cuda" else 4

# Clean transcript, remove any names(of interviewers and interviewees) and stop words/artifacts
SPEAKER_PATTERN = re.compile(r"^\s*([A-Za-z]{1,20}):\s*(.*)$")
TEXT_SKIPS = {
    "AAHP", "Samuel", "Joel", "241", "PO", "Gainesville,", "(352)",
    "https://oral.history.ufl.edu", "African", "Abstract:", "Keywords:",
    "For", "Interviewee:", "Transcribed",
}
TRANSCRIPT_ARTIFACT = re.compile(
    r"""
    ^\s*(?:
        \[(?:laughter|inaudible|unintelligible|crosstalk|pause|
             telephone\s*rings?|phone\s*rings?|recording\s*stops?|
             recording\s*resumes?)[^\]]*\]
    )
    \s*$""",
    flags=re.IGNORECASE | re.VERBOSE,
)
# Also Ignores all names of interviewers/interviewees
PERSON_NAMES = {
    "morini", "proctor", "sandra", "romero", "janet", "ortiz", "rosa",
    "williams", "gwer", "isaac", "chandler", "mattie", "ann", "skinner",
    "mack", "mizell", "oliver", "jones", "christine", "holmes", "arnold",
    "mitchell", "cohen", "marna", "weston", "larnell", "vickers", "ethel",
    "laura", "reaves", "scott", "virgil", "hayes", "atkins", "warren",
    "juanita", "albert", "deloris", "johnson", "david", "richardson",
    "brenda", "washington", "linda", "butler", "margaret", "sharpe",
    "patricia", "burnett", "alvin", "bernard", "hicks", "earl", "jeff",
    "mcmeekin", "mary", "moore", "evelyn", "mickle", "george", "allen",
    "gerald", "jerome", "johncyna", "mcrae", "lee", "bailey", "reuben",
    "brigety", "stephan", "mackey", "andrew", "jean", "chalmers", "willie",
    "mayberry", "mildred", "hilllubin", "thomas", "coward", "samuel",
    "stafford", "pedro", "alonzo", "felder", "hilliard", "nunn", "ray",
    "eberling", "betty", "stewartfullwood", "gwendolyn", "zoharah",
    "simmons", "doris", "manning", "sherry", "dupree", "sherrod", "joel",
    "buchanan", "jessie", "alicia", "antone", "clarence", "pollard",
    "isaiah", "branton", "dixie", "deborah", "moss", "retha", "mae",
    "foxworth", "jimmy", "bobbitt", "yvonne", "hinson", "jerricka",
    "gunter", "mccluney", "diana", "bell", "keith", "yarbrough",
    "yarborough", "westanna", "bobbit", "stephanie", "sam", "taylor",
    "tonyaa", "weathersbee", "kristen", "carlton", "gant", "carnel",
    "jettie", "henderson", "hazel", "gordon", "alma", "russ", "jeraldine",
    "williamsshaw", "mable", "lock", "myrtle", "burks", "helen", "spells",
    "nelson", "janice", "mcmillan", "carey", "godwin", "maple", "smith",
    "martin", "geraldine", "alice", "willette", "walker", "richard",
    "treva", "pittman", "sarah", "mccray", "hall", "estic", "rollings",
    "horace", "mcleod", "drew", "frazier", "hightower", "lillie",
    "tinsley", "william", "monroe", "arthur", "flora", "underwood",
    "walter", "bishop", "diane", "bollet", "lawrence", "hughes", "chester",
    "demps", "kenneth", "dennis", "gail", "wright", "kenny", "sanders",
    "dunwoody", "jay", "thelma", "newberry", "edwina", "carolyn",
    "mickens", "glenn", "lottie", "brown", "huntley", "jackie", "ayers",
    "james", "michelle", "miller", "eugene", "pettis", "joseph", "mccloud",
    "jacquelyn", "tara", "maria", "golden", "rachael", "nickie", "josey",
    "andre", "gainey", "kenya", "mcclain", "ellis", "cusseaux", "henry",
    "lewis", "katesha", "riley", "cydney", "hargro", "kelvin", "dwayne",
    "shaw", "cranford", "ronald", "coleman", "steven", "sikita", "goodrich",
    "michael", "ashby", "mcgill", "madeline", "gervine", "tamya", "welch",
    "beatrice", "certain", "roger", "king", "herbert", "daniel", "marie",
    "calhoun", "charles", "cassandra", "davis", "anna", "baines",
    "verdell", "robinson", "glynnell", "presley", "bernice", "martha",
    "harris", "john", "mayo", "dessie", "islar", "yasmin", "small",
    "elsa", "frederic", "monica", "fay", "gladys", "portia", "emory",
    "palmer", "rawls", "thompson", "cornelius", "clayton", "scherwin",
    "ceola", "watkins", "nkwanda", "jah", "goodwyn", "randy", "klemm",
    "haridelle", "bright", "alethia", "alford", "jane", "adams", "adam",
    "leitha", "nichols", "rebecca", "nathaniel", "sims", "melverine",
    "morris", "jenkins", "wade", "valara", "petteway", "brittany", "ellen",
    "jordan", "janie", "mcclellan", "myrick", "clara", "griffin", "mamie",
    "leath", "curry", "hubert", "elijah", "virginia", "levonia", "eyvonne",
    "andrews", "belinda", "elaine", "daniels", "sharon", "burney", "byran",
    "flagler", "kali", "blount", "johnny", "fair", "claudia", "jan",
    "lawson", "lois", "booker", "dwayitan", "pauline", "franklin",
    "claretha", "bradley", "frances", "wilson", "bettie", "blakely",
    "lurie", "brian", "favors", "lakay", "banks", "hermia", "sherman",
    "whitney", "battlebaptiste", "rosemary", "barnett", "lowery",
    "lexington", "blair", "ella", "driskell", "kimbrough", "padgett",
    "cohens", "byllye", "avery", "perkins", "price", "armbrister",
    "jason", "yulee", "vendarae", "corbett", "freddie", "hickmon",
    "dolores", "mccullough", "jefferson", "rogers", "rufus", "brooks",
    "barbara", "towns", "antoinette", "vonda", "pearline", "matthis",
    "rickman", "eula", "arago", "zelphia", "chambers", "bertha", "abungu",
    "jordon", "leonard", "young", "jocelyn", "carter", "ingram",
    "shirleyjo", "tuffs", "regennia", "hannibal", "vivian", "carrington",
    "glover", "johnetta", "betsch", "cole", "lorene", "clementine",
    "hibbert", "wayne", "fields", "peggy", "brunache", "shawn", "bryant",
    "marva", "murray", "clenton", "cannion", "jasmine", "maya", "rose",
    "marshall", "angenetta", "durrant", "viola", "cora", "tyson", "howard",
    "armour", "shirley", "schumpert", "ferman", "parker", "ryan", "hills",
    "peterson", "douglas", "sollie", "pinkston", "dorsey", "joe", "eddie",
    "harmeling", "noabstract", "rhonda", "earnestine", "roberta",
    "stephens", "priscilla", "kruize", "martine", "taffany", "fisher",
    "johnie", "rackard", "malik", "rahim", "larry", "saunders", "chestnut",
    "iii", "frederick", "carrie", "shepherd", "nikki", "giovanni", "maye",
    "julien", "louissteen", "cummings", "olga", "luresa", "lake", "carol",
    "greenlee", "leon", "leoris", "stanley", "janquez", "hattie",
    "castell", "esther", "carl", "edward", "irvin", "anthony", "major",
    "alberta", "hamilton", "mason", "rachel", "shelley", "clyburn",
    "rainey", "reid", "jesse", "jetson", "grimes", "hunter", "julian",
    "beverly", "moreland", "lumpkin", "harvey", "prevell", "barber",
    "trevor", "harvin", "gilbert", "wendell", "patrick", "fannie",
    "mcdougal", "cusick", "ernest", "sneed", "boatwright", "mickey",
    "michaux", "kathleen", "cleaver", "gainous", "mcivory", "vanessa",
    "bonner", "alena", "vinson", "lopez", "oscar", "worthy", "yves",
    "vaughan", "ashley", "marceus", "roberts", "sophia", "threat",
    "kitty", "gallon", "ernestine", "dave", "gussie", "faye", "leroy",
    "seabrooks", "lenard", "panzie", "rafe", "elmer", "norris", "hunt",
    "bernadette", "cailler", "goston", "nikitah", "okenbera", "imani",
    "horne", "josephine", "spearman", "levy", "antonette", "bennett",
    "mavis", "agbandjemckenna", "stephen", "barrington", "anderson",
    "duchess", "austin", "marion", "warford", "teresa", "ferrell",
    "madison", "akil", "reynolds", "flavius", "wyard", "earsel",
    "estelle", "forehand", "stevens", "reed", "crenshaw", "katrina",
    "rolle", "bessie", "nina", "jacob", "luke", "nickson", "pat",
    "mccutcheon", "fox", "savannah", "campbell", "ida", "toney", "lovie",
    "wells", "simms", "ebony", "bonaparte", "essie", "kenneth", "victoria",
    "booth", "elizabeth", "durant", "sallie", "hollis", "tameka", "hobbs",
    "noesha", "mariah", "noel", "rollins", "filer", "kevin", "graham",
    "kiora", "whittle", "kenney", "keirten", "nivol", "cunningham",
    "turner", "cecile", "scoon", "eva", "mannings", "whitfield", "steele",
    "woodard", "lorenzo", "edwards", "cadets", "bobby", "perry", "marker",
    "stevenson", "dorothy", "lafanette", "soleswoods", "powell", "barry",
    "bickham", "georgia", "sunday", "tellis", "veasley", "minor",
    "eurydice", "ieshia", "watson", "melvin", "webb", "hixon", "frankie",
    "mcintosh", "sylvia", "todd", "maggie", "wiggins", "townsend",
    "reginald", "foxx", "kemberly", "ronnie", "ned", "ransom", "mckinney",
    "harrison", "masseyharpoole", "brickler", "brady", "vogt", "allonia",
    "demetric", "witchell", "lafortune", "vi", "regis", "rebia", "berry",
    "frank", "jennifer", "thelusma", "land", "bowser", "haskins", "cottie",
    "aurora", "martinez", "adlancy", "osborn", "bessy", "bradshaw",
    "robbie", "gregg", "preer", "robbins", "gross", "alexander", "francis",
    "tolbert", "fowlkes", "moultry", "wallace", "edgar", "osmond",
    "sharpless", "christopher", "busey", "canton", "watts", "carlos",
    "alvarez", "paul",
}

# Candidate topics, guides the model towards these topics
ZERO_SHOT_CANDIDATES = [
    "civil rights and racial discrimination",
    "education and school",
    "work and employment",
    "farming and agriculture",
    "family and home",
    "government and voting",
    "health and medicine",
    "religion and church",
    "military and war",
    "music and arts",
    "sports and athletics",
    "police and law enforcement",
]
TOPIC_DESCRIPTIONS = {
    "civil rights and racial discrimination": (
        "Experiences of racial discrimination, segregation, civil rights, "
        "racial inequality, protest, integration, or Black political activism."
    ),
    "education and school": (
        "Experiences involving schools, teachers, students, classes, education, "
        "colleges, universities, learning, or segregation in education."
    ),
    "work and employment": (
        "Experiences involving jobs, employment, businesses, occupations, "
        "workplaces, wages, promotions, trades, or financial circumstances."
    ),
    "farming and agriculture": (
        "Experiences involving farms, crops, livestock, agricultural labor, "
        "land cultivation, or rural food production."
    ),
    "family and home": (
        "Experiences involving parents, children, spouses, marriage, relatives, "
        "household life, family relationships, or the home."
    ),
    "government and voting": (
        "Experiences involving government, voting, elections, legislation, "
        "democracy, representation, or civic participation."
    ),
    "health and medicine": (
        "Experiences involving illness, injury, disability, doctors, hospitals, "
        "medical treatment, healthcare, or recovery."
    ),
    "religion and church": (
        "Experiences involving churches, worship, prayer, faith, religious "
        "beliefs, clergy, congregations, or religious communities."
    ),
    "military and war": (
        "Experiences involving military service, armed forces, war, combat, "
        "deployment, veterans, military training, or veterans' services."
    ),
    "music and arts": (
        "Experiences involving music, singing, musicians, instruments, "
        "performances, bands, visual arts, or cultural expression."
    ),
    "sports and athletics": (
        "Experiences involving sports, athletics, teams, games, competition, "
        "coaches, or recreational activities."
    ),
    "police and law enforcement": (
        "Experiences involving police, arrest, law enforcement, courts, jail, "
        "prison, or incarceration."
    ),
}

stopwords = list(
    text.ENGLISH_STOP_WORDS
    .union(PERSON_NAMES)
    .union({
        "including", "interview", "interviews", "like", "rev", "ms",
        "mrs", "dr", "mr", "didn", "interviewed", "aahp", "oh", "good",
        "know", "said", "says", "asked", "told", "program", "nebo",
        "jerkins", "africanamericanoralhistory", "right", "maybe", "19",
        "50s", "60s", "oral", "end", "born", "don",
    })
)


def clean_sentence(sentence):
    sentence = re.sub(r"^\s*[A-Za-z]{1,20}:\s*", "", sentence)
    sentence = re.sub(r"\s+", " ", sentence)
    return sentence.strip()


def is_usable_sentence(sentence):
    clean = clean_sentence(sentence)
    if TRANSCRIPT_ARTIFACT.search(clean):
        return False
    words = re.findall(r"\b[\w''-]+\b", clean)
    if len(re.findall(r"[A-Za-z]", clean)) < 15:
        return False
    return True


# Load the Trancripts to read all of the sentences, so that they can be fed as docs to the model
if not TRANSCRIPT_DIR.exists():
    sys.exit(f"ERROR: Transcript directory '{TRANSCRIPT_DIR}' not found.")

transcript_paths = sorted(TRANSCRIPT_DIR.glob("*.txt"))
if not transcript_paths:
    sys.exit(f"ERROR: No txt files found in '{TRANSCRIPT_DIR}'.")

docs: list[str] = []
doc_sources: list[str] = []

for transcript_path in transcript_paths:
    if len(docs) >= MAX_SENTENCES:
        break

    # Keep track of the current speaker, this makes in possible to seperate the interviewee and interviewer lines
    current_speaker = None
    file_sentences = 0

    with open(transcript_path, "r", encoding="utf-8") as f:
        for line in f:
            if len(docs) >= MAX_SENTENCES:
                break

            line = line.rstrip("\n").strip()
            if not line:
                continue

            first_word = line.split()[0] if line.split() else ""
            if first_word in TEXT_SKIPS:
                continue

            speaker_match = SPEAKER_PATTERN.match(line)
            if speaker_match:
                current_speaker = speaker_match.group(1).upper()
                line = speaker_match.group(2).strip()

            for sentence in sent_tokenize(line):
                if not is_usable_sentence(sentence):
                    continue
                docs.append(clean_sentence(sentence))
                doc_sources.append(transcript_path.stem)
                file_sentences += 1

print(f"\nTotal sentences : {len(docs):,}")
print(f"Interviews      : {len(set(doc_sources))}")

# Configure the BERTopic model pipeline
embedding_model_gpu = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=DEVICE)
if DEVICE == "cuda":
    embedding_model_gpu.half()

embeddings = embedding_model_gpu.encode(docs, batch_size=EMBEDDING_BATCH, show_progress_bar=True, convert_to_numpy=True,
    normalize_embeddings=True,)

# BERTopic needs a cpu reference to the model
embedding_model_ref = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")


umap_model = UMAP(n_neighbors=25, n_components=5, min_dist=0.0, metric="cosine", random_state=42, n_epochs=500,
                  low_memory=False)

hdbscan_model = HDBSCAN(min_cluster_size=145, min_samples=20, metric="euclidean", cluster_selection_method="eom",
                        prediction_data=True, core_dist_n_jobs=-1)

vectorizer_model = CountVectorizer(stop_words=stopwords, ngram_range=(1, 2), min_df=2, max_df=0.85)

representation_model = KeyBERTInspired(top_n_words=15)

ctfidf_model = ClassTfidfTransformer(seed_multiplier=10, bm25_weighting=True, reduce_frequent_words=True)

# Create the model with the configurations
topic_model = BERTopic(
    embedding_model=embedding_model_ref,
    hdbscan_model=hdbscan_model,
    umap_model=umap_model,
    vectorizer_model=vectorizer_model,
    representation_model=representation_model,
    ctfidf_model=ctfidf_model,
    zeroshot_topic_list=ZERO_SHOT_CANDIDATES,
    nr_topics="auto",
    top_n_words=15,
    verbose=True,
)

# Run the model on the sentences(docs) with the (gpu) embedding model
topics, _ = topic_model.fit_transform(docs, embeddings=embeddings)

topic_info = topic_model.get_topic_info()
n_topics = len([t for t in topic_info["Topic"] if t != -1])
n_outliers = sum(1 for t in topics if t == -1)

print(f"\nTopics found: {n_topics}")

print("\nTop keywords per topic:")
for t in sorted(topic_info["Topic"]):
    if t == -1:
        continue
    keywords = [word for word, _ in topic_model.get_topic(t)[:6]]
    size = topic_model.get_topic_freq(t)
    print(f"  Topic {t:>3} ({size:>5} docs): {', '.join(keywords)}")

# Find and clear the model path
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
if MODEL_PATH.is_file():
    MODEL_PATH.unlink()
elif MODEL_PATH.is_dir():
    import shutil
    shutil.rmtree(MODEL_PATH)

# Save the BERTopic model pickle(makes it easier for the sentiment analysis to load the model).
topic_model.save(str(MODEL_PATH), serialization="pickle", save_ctfidf=True,
                 save_embedding_model="sentence-transformers/all-mpnet-base-v2")

print(f"Training complete. Model loaded to {MODEL_PATH}")
