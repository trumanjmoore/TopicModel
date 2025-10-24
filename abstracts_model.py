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

candidate_seed_topics = [
        ["civil rights", "march", "freedom", "justice", "activist", "protest", "demonstration", "equality", "movement",
         "struggle", "rights", "advocacy", "boycott", "nonviolence", "rally", "petition", "community", "fight", "voice",
         "change"],

        ["education", "school", "teacher", "student", "college", "university", "campus", "classroom", "learning",
         "books",
         "curriculum", "graduate", "degree", "professor", "study", "lesson", "library", "scholarship", "academic",
         "literacy"],

        ["job", "work", "employment", "career", "wages", "money", "income", "pay", "occupation", "labor", "factory",
         "opportunity", "office", "union", "industry", "economy", "market", "salary", "business", "position"],

        ["farm", "agriculture", "crops", "land", "harvest", "soil", "plant", "season", "rural", "livestock", "field",
         "gardening", "farmer", "produce", "orchard", "seed", "tractor", "barn", "plantation", "food"],

        ["family", "children", "parents", "home", "kin", "siblings", "relatives", "mother", "father", "grandparents",
         "household", "bond", "care", "love", "support", "generation", "spouse", "cousins", "community", "domestic"],

        ["government", "policy", "law", "election", "vote", "democracy", "officials", "congress", "senate",
         "constitution",
         "rights", "representation", "justice", "legislation", "state", "federal", "authority", "public", "citizen",
         "governor"],

        ["health", "doctor", "hospital", "medicine", "care", "nurse", "illness", "sick", "disease", "treatment",
         "patient", "clinic", "wellness", "injury", "mental", "therapy", "surgery", "prevention", "healthcare", "cure"],

        ["african american", "black history", "heritage", "culture", "identity", "belonging", "tradition", "community",
         "ancestry", "pride", "diaspora", "freedom", "race", "contribution", "struggle", "legacy", "expression",
         "roots", "resilience", "empowerment"],

        ["segregation", "separate", "racism", "discrimination", "jim crow", "apart", "color", "inequality", "prejudice",
         "divide", "exclusion", "bias", "segregated", "desegregation", "rights", "barrier", "oppression", "integration",
         "restrict", "denial"],

        ["age", "old", "young", "generation", "years", "youth", "elderly", "senior", "childhood", "teen", "adult",
         "growth", "birthday", "life", "stage", "mature", "experience", "time", "decade", "aging"],

        ["slavery", "enslaved", "slave", "plantation", "freedom", "abolition", "bondage", "chains", "auction", "labor",
         "oppression", "servitude", "cotton", "whip", "owner", "forced", "trade", "human", "captivity", "emancipation"],

        ["religion", "church", "faith", "worship", "spiritual", "belief", "bible", "preacher", "pastor", "god",
         "prayer", "sermon", "service", "hymn", "congregation", "gospel", "holy", "ritual", "temple", "devotion"],

        ["military", "army", "service", "war", "veteran", "battle", "soldier", "uniform", "duty", "training",
         "weapon", "conflict", "draft", "camp", "march", "deployment", "victory", "rank", "base", "combat"],

        ["music", "song", "band", "jazz", "art", "painting", "artist", "creative", "melody", "dance",
         "instrument", "choir", "performance", "singing", "culture", "poetry", "guitar", "blues", "expression",
         "festival"],

        ["sports", "game", "team", "football", "basketball", "baseball", "athlete", "competition", "play", "score",
         "field", "coach", "training", "stadium", "league", "championship", "track", "player", "victory", "match"],

        ["migration", "move", "relocate", "immigrate", "travel", "journey", "departure", "arrival", "emigrate",
         "settle",
         "resettle", "displacement", "shift", "transit", "border", "region", "diaspora", "community", "movement",
         "change"],

        ["police", "officer", "law enforcement", "arrest", "patrol", "housing", "home", "rent", "neighborhood",
         "safety",
         "justice", "crime", "authority", "court", "order", "sheriff", "guard", "law", "jail", "violence"]
    ]


# load all the interview transcripts
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
print(len(docs))

umap_model = UMAP(n_components=2, metric='cosine', random_state=42)

hdbscan_model = HDBSCAN(min_cluster_size=5,  min_samples=5, metric='euclidean', cluster_selection_method='eom', prediction_data=True)

representation_model = KeyBERTInspired()

names = ['janet', 'paul', 'ortiz', 'rosa', 'b', 'williams', 'gwer', 'panel', 'isaac', 'chandler', 'jr', 'isaac', 'chandler', 'jr', 'mattie', 'williams', 'ann', 'skinner', 'mack', 'mizell', 'oliver', 'jones', 'christine', 'holmes', 'christine', 'holmes', 'arnold', 'mitchell', 'arnold', 'mitchell', 'paul', 'cohen', 'marna', 'weston', 'larnell', 'vickers', 'ethel', 'williams', 'laura', 'reaves', 'laura', 'reaves', 'laura', 'scott', 'reaves', 'virgil', 'hayes', 'atkins', 'warren', 'juanita', 'scott', 'williams', 'albert', 'white', 'deloris', 'johnson', 'david', 'richardson', 'isaac', 'jones', 'brenda', 'washington', 'linda', 'butler', 'margaret', 'sharpe', 'patricia', 'burnett', 'alvin', 'butler', 'bernard', 'hicks', 'earl', 'williams', 'jeff', 'mcmeekin', 'mary', 'moore', 'evelyn', 'mickle', 'w', 'george', 'allen', 'gerald', 'jerome', 'johnson', 'johncyna', 'mcrae', 'lee', 'bailey', 'reuben', 'brigety', 'stephan', 'mickle', 'bernard', 'mackey', 'andrew', 'mickle', 'jean', 'chalmers', 'rosa', 'williams', 'evelyn', 'marie', 'moore', 'mickle', 'evelyn', 'marie', 'moore', 'mickle', 'willie', 'mayberry', 'mildred', 'hilllubin', 'mildred', 'hilllubin', 'thomas', 'coward', 'samuel', 'stafford', 'marna', 'weston', 'paul', 'pedro', 'ortiz', 'alonzo', 'felder', 'alonzo', 'felder', 'alonzo', 'felder', 'patricia', 'hilliard', 'nunn', 'paul', 'andrew', 'ortiz', 'ray', 'eberling', 'betty', 'stewartfullwood', 'gwendolyn', 'zoharah', 'simmons', 'gwendolyn', 'zoharah', 'simmons', 'gwendolyn', 'zoharah', 'simmons', 'd', 'gwendolyn', 'zoharah', 'simmons', '_', 'd', 'gwendolyn', 'zoharah', 'simmons', 'doris', 'manning', 'sherry', 'dupree', 'sherry', 'sherrod', 'dupree', 'sherry', 'dupree', 'joel', 'buchanan', 'joel', 'buchanan', 'speech', 'jessie', 'jones', 'alicia', 'antone', 'clarence', 'pollard', 'isaiah', 'branton', 'laura', 'dixie', 'deborah', 'moss', 'retha', 'mae', 'foxworth', 'jimmy', 'bobbitt', 'yvonne', 'hinson', 'jerricka', 'gunter', 'warren', 'mccluney', 'diana', 'bell', 'keith', 'yarbrough', 'keith', 'yarborough', 'westanna', 'bobbit', 'stephanie', 'pollard', 'sam', 'taylor', 'tonyaa', 'weathersbee', 'kristen', 'yarborough', 'carlton', 'gant', 'carnel', 'and', 'jettie', 'henderson', 'hazel', 'gordon', 'alma', 'russ', 'jeraldine', 'williamsshaw', 'mable', 'lock', 'myrtle', 'burks', 'helen', 'spells', 'nelson', 'janice', 'mcmillan', 'carey', 'godwin', 'mary', 'maple', 'smith', 'martin', 'geraldine', 'smith', 'alice', 'hayes', 'willette', 'walker', 'richard', 'black', 'treva', 'walker', 'jimmy', 'pittman', 'sarah', 'mccray', 'hall', 'estic', 'rollings', 'horace', 'mcleod', 'horace', 'mcleod', 'drew', 'frazier', 'willie', 'hightower', 'lillie', 'tinsley', 'william', 'monroe', 'arthur', 'thomas', 'flora', 'underwood', 'walter', 'bishop', 'diane', 'taylor', 'bollet', 'lawrence', 'hughes', 'jr', 'chester', 'demps', '_', 'kenneth', 'dennis', 'gail', 'holmes', 'walter', 'wright', 'kenny', 'sanders', 'deloris', 'dunwoody', 'jay', 'thelma', 'walker', 'newberry', 'betty', 'white', 'hughes', 'edwina', 'moore', 'carolyn', 'mickens', 'glenn', 'lottie', 'mae', 'white', 'brown', 'mary', 'huntley', 'gant', 'jackie', 'ayers', 'james', '_', 'michelle', 'miller', 'eugene', 'pettis', 'clarence', 'brown', 'joseph', 'mccloud', 'jacquelyn', 'williams', 'jones', 'tara', 'miller', 'maria', 'golden', '_', 'alicia', 'golden', 'rachael', 'nickie', 'jerome', 'josey', 'andre', 'gainey', 'kenya', 'mcclain', 'ellis', 'allen', 'cusseaux', 'henry', 'lewis', 'katesha', 'riley', '_', 'cydney', 'hargro', 'kelvin', 'henry', 'dwayne', 'shaw', 'cranford', 'ronald', 'coleman', 'jr', 'cranford', 'ronald', 'coleman', 'jr', 'steven', '_', 'sikita', 'goodrich', 'michael', 'ashby', 'gerald', 'mcgill', 'madeline', 'gervine', 'mccloud', 'tamya', 'smith', 'and', 'others', 'thelma', 'welch', 'beatrice', 'certain', 'roger', 'king', 'roger', 'king', 'herbert', 'jones', 'daniel', 'gainey', 'marie', 'calhoun', 'charles', 'moore', 'cassandra', 'davis', 'anna', 'baines', 'verdell', 'robinson', 'glynnell', 'presley', 'bernice', 'presley', 'martha', 'harris', 'john', 'mayo', 'dessie', 'robinson', 'mae', 'islar', 'mae', 'islar', 'yasmin', 'small', 'elsa', 'frederic', 'monica', 'smith', 'thomas', 'fay', 'gladys', 'wright', 'portia', 'taylor', 'emory', 'harris', 'carolyn', 'palmer', 'john', 'rawls', 'madeline', 'thompson', 'cornelius', 'clayton', 'jr', 'cornelius', 'clayton', 'scherwin', 'henry', 'ceola', 'watkins', 'nkwanda', 'jah', 'lawrence', 'goodwyn', 'randy', 'klemm', 'haridelle', 'taylor', 'bright', 'alethia', 'brown', 'florida', 'alford', 'jane', 'adams', 'leitha', 'nichols', 'rebecca', 'hall', 'and', 'nathaniel', 'hall', 'yvonne', 'robinson', 'deborah', 'sims', 'melverine', 'morris', 'mary', 'alice', 'jenkins', 'henry', 'wade', 'valara', 'petteway', 'brittany', 'o_neil', 'charles', 'demps', 'ellen', 'jordan', 'janie', 'williams', 'richard', 'mcclellan', 'mary', 'lee', 'myrick', 'isaac', 'jones', 'clara', 'griffin', 'mamie', 'lee', 'leath', 'mamie', 'lee', 'leath', 'mamie', 'lee', 'leath', 'patricia', 'curry', 'hubert', 'curry', 'elijah', 'lewis', 'virginia', 'hayes', 'levonia', 'king', 'paul', 'ortiz', 'gladys', 'thompson', 'eyvonne', 'andrews', 'ceola', 'watkins', 'belinda', 'elaine', 'daniels', 'sharon', 'burney', 'sharon', 'burney', 'george', 'washington', 'byran', 'williams', 'lewis', 'flagler', 'kali', 'blount', 'johnny', 'fair', 'claudia', 'rawls', 'jan', 'lawson', 'lois', 'harris', 'booker', 'dwayitan', 'pauline', 'lawrence', 'martha', 'franklin', 'claretha', 'bradley', 'frances', 'wilson', 'gwendolyn', 'williams', 'bettie', 'blakely', 'bettie', 'blakely', 'mary', 'hall', 'daniels', 'lurie', 'and', 'brian', 'favors', 'lakay', 'banks', 'isaiah', 'branton', 'hermia', 'sherman', 'whitney', 'battlebaptiste', 'rosemary', 'williams', 'and', 'mildred', 'barnett', 'jerome', 'mack', '_', 'juanita', 'mack', 'linda', 'lowery', 'lexington', 'blair', 'ella', 'mae', 'driskell', 'linda', 'kimbrough', 'david', 'padgett', 'robert', 'coleman', 'carolyn', 'cohens', 'byllye', 'avery', 'gladys', 'perkins', 'gladys', 'perkins', 'lee', 'j', 'price', 'james', 'b', 'miller', 'hazel', 'armbrister', 'jason', 'yulee', 'vendarae', 'lewis', 'jordan', 'corbett', 'eugene', 'gainey', 'freddie', 'hickmon', 'dolores', 'mccullough', 'jefferson', 'rogers', 'jefferson', 'rogers', 'henry', 'leath', '_', 'mattie', 'leath', 'rufus', 'brooks', 'barbara', 'smith', 'cornelius', 'towns', 'isaiah', 'branton', 'antoinette', 'jackson', 'vonda', 'richardson', 'pearline', 'jones', 'jw', 'welch', 'and', 'matthis', 'harris', 'ray', 'rickman', 'eula', 'harris', 'matthis', 'harris', 'arago', 'welch', 'joseph', 'welch', 'zelphia', 'chambers', 'bill', 'white', 'joseph', 'welch', 'bertha', 'lee', 'george', 'abungu', 'jordon', 'corbett', 'leonard', 'and', 'alonzo', 'young', 'jocelyn', 'carter', 'ingram', 'shirleyjo', 'tuffs', 'regennia', 'williams', 'hannibal', 'square', 'heritage', 'center', 'panel', 'vivian', 'carrington', 'arthur', 'glover', 'johnetta', 'betsch', 'cole', 'lorene', 'smith', 'clementine', 'hibbert', 'fe', 'wayne', 'fields', 'peggy', 'brunache', 'shawn', 'fields', 'charles', 'bryant', 'mary', 'bryant', 'ocala', 'hunting', 'and', 'fishing', 'club', 'mary', 'and', 'van', 'banks', 'marva', 'murray', 'nathaniel', 'harris', 'clara', 'smith', 'clenton', 'taylor', 'willie', 'cannion', 'jasmine', 'and', 'maya', 'jordan', 'rose', 'marshall', 'angenetta', 'durrant', 'and', 'viola', 'franklin', 'lois', 'miller', 'cora', 'tyson', 'matthis', 'harris', 'leonard', 'and', 'mary', 'reynolds', 'howard', 'armour', 'shirley', 'butler', 'betty', 'schumpert', 'johnson', 'and', 'lee', 'ferman', 'welch', 'parker', 'and', 'moore', 'jackson', 'ryan', 'hills', 'daniel', 'gainey', 'smith', 'peterson', 'king', 'douglas', 'and', 'carter', 'sollie', 'mitchell', 'ann', 'pinkston', 'town', 'meeting', 'african', 'american', 'dorsey', 'miller', 'joe', 'eddie', 'scott', 'dan', 'harmeling', 'dan', 'harmeling', 'robert', 'hall', 'may', 'stafford', 'noabstract', 'retha', 'mae', 'cooks', 'rhonda', 'johnson', 'earnestine', 'johnson', 'roberta', 'stephens', 'charles', 'washington', 'priscilla', 'kruize', 'bernard', 'williams', 'martine', 'young', 'taffany', 'brown', 'rosa', 'lee', 'fisher', 'robert', '_teddy', 'bear_', 'marshall', 'johnie', 'moore', 'william', 'james', 'william', 'james', 'david', 'rackard', 'malik', 'rahim', 'larry', 'saunders', 'charles', 'chestnut', 'iii', 'frederick', 'pinkston', 'carrie', 'bell', 'jones', 'women', 'leaders', 'exhibit', 'and', 'rosewood', 'exhibit', 'vivian', 'shepherd', 'nikki', 'giovanni', 'maye', 'st', 'julien', 'louissteen', 'cummings', 'olga', 'mitchell', 'richard', 'hall', 'luresa', 'lake', 'carol', 'greenlee', 'leon', 'and', 'cora', 'west', 'gail', 'jones', 'leoris', 'richardson', 'stanley', 'richardson', 'janquez', 'west', 'hattie', 'castell', '_', 'olga', 'mitchell', 'esther', 'thomas', 'carl', 'calhoun', 'edward', 'irvin', 'anthony', 'major', 'alberta', 'brown', 'hamilton', 'carolyn', 'mason', 'rachel', 'shelley', 'thomas', 'clyburn', 'ufdc_', 'eddie', 'l', 'rainey', 'edward', 'james', 'iii', 'ethel', 'reid', 'hayes', 'gwendolyn', 'atkins', 'helen', 'dixon', 'jesse', 'johnson', 'jetson', 'grimes', 'johnny', 'hunter', 'julian', 'and', 'beverly', 'moreland', 'kelvin', 'lumpkin', 'mary', 'alice', 'simmons', 'nathaniel', 'harvey', 'prevell', 'barber', 'robert', 'l', 'taylor', 'trevor', 'd', 'harvey', 'wade', 'harvin', 'walter', 'l', 'gilbert', 'iii', 'wendell', 'patrick', 'carter', 'willie', 'charles', 'shaw', 'fannie', 'mcdougal', 'james', 'cusick', 'ernest', 'sneed', 'john', 'boatwright', 'mickey', 'michaux', 'kathleen', 'cleaver', 'jones,', 'gainous,', 'mcivory', 'james', 'brown', '_', 'vanessa', 'bonner', 'alena', 'lawson', 'freddie', 'hickmon', 'horace', 'moore', 'jancie', 'vinson', 'roberta', 'lopez', 'yvonne', 'hinson', 'rawls', 'oscar', 'sam', 'harris', 'oscar', 'sam', 'harris', 'james', 'worthy', 'yves', 'vaughan', 'ashley', 'marceus', 'steven', 'roberts', 'sophia', 'threat', 'kitty', 'oliver', 'john', 'nelson', 'john', 'nelson', 'dennis', 'gallon', 'warren', 'henderson', 'ernestine', 'dave', 'gussie', 'butler', 'gussie', 'butler', 'faye', 'williams', 'cornelius', 'davis', 'leroy', 'seabrooks', 'jr', 'lenard', 'davis', 'panzie', 'parker', 'rafe', 'johnson', 'rafe', 'johnson', 'elmer', 'henry', 'barbara', 'norris', 'hunt', 'davis', 'jr', 'bernadette', 'cailler', 'gordon', 'carey', 'mary', 'williams', 'charles', 'goston', 'frederick', 'fisher', 'nikitah', 'okenbera', 'imani', 'nikitah', 'imani', 'david', 'horne', 'josephine', 'bryant', 'leonard', 'spearman', 'jr', 'sharon', 'glover', 'vincent', 'green', 'hazel', 'levy', 'antonette', 'bennett', 'mavis', 'agbandjemckenna', 'stephen', 'roberts', 'eddie', 'barrington', 'walter', 'anderson', 'duchess', 'harris', 'sharon', 'austin', 'sharon', 'austin', 'marion', 'coleman', 'dr', 'john', 'warford', 'teresa', 'brown', 'william', 'ferrell', 'arthur', 'lee', 'madison', 'akil', 'reynolds', 'akil', 'reynolds', 'flavius', 'johnson', 'wyard', 'dennis', 'earsel', 'lewis', 'earsel', 'lewis', 'earsel', 'lewis', 'linda', 'dixie', 'estelle', 'forehand', 'betty', 'stevens', 'jc', 'reed', 'randy', 'adams', 'anthony', 'crenshaw', 'katrina', 'rolle', 'isaac', 'anderson', 'bessie', 'washington', 'russel', 'brown', 'nina', 'rogers', 'jacob', 'johnson', 'jacob', 'johnson', 'luke', 'black', 'mae', 'ola', 'nickson', 'pat', 'mccutcheon', 'kathleen', 'fox', 'savannah', 'campbell', 'pat', 'mccutcheon', 'william', 'atkins', 'ida', 'mae', 'golden', 'hubert', 'toney', 'sr', 'george', 'mayo', 'lovie', 'wells', 'jr', 'simms', 'and', 'simms', 'simms', 'and', 'simms', 'mattie', 'harris', 'mattie', 'young', 'gwendolyn', 'brooks', 'frederick', 'white', 'walt', 'wesley', 'sr', '_', 'marion', 'redden', 'sims', 'ronald', 'davis', 'ebony', 'thomas', 'john', 'bonaparte', 'john', 'johnson', 'essie', 'anderson', 'kenneth', 'green', 'victoria', 'jones', 'john', 'booth', '_', 'elizabeth', 'durant', 'elizabeth', 'durant', 'jenkins', 'and', 'jenkins', 'emanuel', 'bridges', 'sallie', 'hollis', 'tameka', 'hobbs', 'noesha', 'mariah', 'noel', 'rollins', 'college', 'vivian', 'filer', 'kevin', 'sharpe', 'dr', 'elizabeth', 'graham', 'kiora', 'whittle', 'mary', 'kenney', '_', 'keirten', 'nivol', 'lee', 'and', 'cunningham', 'ashley', 'marceus', 'nathaniel', 'turner', 'monroe', 'lee', 'cecile', 'scoon', 'eva', 'mannings', 'robert', 'brown', 'jr', 'whitfield', 'jenkins', '_', 'bus', 'boycott', 'steele', 'and', 'richardson', 'ab', 'copy', 'barbara', 'higgins', 'bqt', '_', 'george', 'woodard', 'lorenzo', 'edwards', 'wp', 'cadets', 'bobby', 'james', 'july', 'perry', 'historic', 'marker', 'bryan', 'stevenson', 'speech', 'dorothy', 'marshall', 'lafanette', 'soleswoods', 'lafanette', 'soleswoods', 'powell', 'and', 'lawrence', 'barry', 'bickham', 'georgia', 'sunday', 'joe', 'davis', 'freddie', 'tellis', 'ronald', 'johnson', 'john', 'veasley', 'coleman', 'and', 'arnold', 'henry', 'minor', 'eurydice', 'stanley', 'ieshia', 'williams', 'copy', 'sam', 'watson', 'melvin', 'turner', 'jr', 'mamie', 'webb', 'hixon', 'frankie', 'mcintosh', 'sylvia', 'todd', 'maggie', 'wilson', 'marilynn', 'wiggins', 'lewis', 'and', 'townsend', 'reginald', 'lewis', 'evelyn', 'foxx', 'eddie', 'thomas', 'kemberly', 'jackson', 'ronnie', 'griffin', 'ned', 'hill', 'jr', 'ben', 'ransom', 'jr', 'henry', 'mckinney', 'jr', 'william', 'harrison', 'deloris', 'masseyharpoole', 'henry', 'steele', 'brickler', '_', 'brickler', 'brady', 'vogt', 'allonia', 'griffin', 'demetric', 'jackson', 'witchell', 'lafortune', 'vi', 'whitfield', 'memorial', 'service', 'newberry', 'six', 'regis', 'boatwright', 'rebia', 'berry', 'frank', 'washington', 'jennifer', 'thelusma', 'hazel', 'land', '_', 'robert', 'bowser', 'kathleen', 'haskins', '_', 'cottie', 'wright', '_', 'aurora', 'martinez', 'adlancy', 'horne', 'eddie', 'lee', 'osborn', 'bessy', 'moore', 'bradshaw', 'joseph', 'holmes', 'robbie', 'gregg', 'ronald', 'preer', 'michael', 'roberts', 'larry', 'robbins', 'robert', 'gross', 'david', 'alexander', 'francis', 'tolbert', 'michael', 'fowlkes', 'james', 'moultry', 'joseph', 'lewis', 'walter', 'wallace', 'edgar', 'carter', 'osmond', 'sharpless', 'christopher', 'busey', 'david', 'canton', 'paul', 'ortiz', 'bertha', 'watts', 'carlos', 'alvarez', 'ray', 'eberling']

stopwords = list(text.ENGLISH_STOP_WORDS.union(['including', 'interview', 'like', 'rev', 'ms', 'mrs', 'dr', 'sherry',
                                                'mr', 'sherrod', 'duppree', 'interviewed', 'interview', 'interviewer',
                                                'AAHP', 'yeah', 'uh', 'huh', 'oh', 'good', 'know', 'said', 'says',
                                                'asked', 'told', 'program', 'nebo', 'narrator', 'interviewee']).union(names))
vectorizer_model = CountVectorizer(stop_words=stopwords)

#embedding_model = Model2VecBackend("sentence-transformers/all-MiniLM-L6-v2",
#    distill=True, distill_kwargs={"pca_dims": 256}, distill_vectorizer=vectorizer_model)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

ctfidf_model = ClassTfidfTransformer(
    seed_words=candidate_seed_topics,
    seed_multiplier=100
)

topic_model = (BERTopic(embedding_model=embedding_model, hdbscan_model=hdbscan_model,
                        vectorizer_model=vectorizer_model, representation_model=representation_model,
                        ctfidf_model=ctfidf_model, seed_topic_list=candidate_seed_topics,
                        verbose=True, nr_topics=20))
topics, probs = topic_model.fit_transform(docs)

hierarchical_topics = topic_model.hierarchical_topics(docs)


print(len(topic_model.get_topics()))
print((topic_model.get_topic(-1)))
print(topic_model.get_topic_freq(-1))

for t in topic_model.get_topic_info()['Topic']:
    if t == -1:
        continue  # skip outliers
    print(f"Topic {t}:")
    print(f"    {topic_model.get_topic(t)}")

fig = topic_model.visualize_topics()
heat_map = topic_model.visualize_heatmap()
fig.show()
heat_map.show()
tree = topic_model.get_topic_tree(hierarchical_topics)
print(tree)

topic_model.save("abstracts_model", serialization="safetensors", save_ctfidf=True, save_embedding_model=embedding_model)



