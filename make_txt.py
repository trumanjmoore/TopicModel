import os
import re

docs = []
names = []
for filename in os.listdir("txt"):
    filepath = os.path.join("txt", filename)
    with open(filepath, 'r', encoding='utf-8') as file:
        text = file.read()
        name = file.name
        name = re.sub(r'[0-9]+', '', name)
        name = re.sub('AAHP', '', name)
        name = re.sub('ufdc.txt', '', name)
        name = re.sub('ufdc', '', name)
        name = re.sub('txt', '', name)
        name = re.sub('\.', '', name)
        name = re.sub('-', '', name)
        name = re.sub('txt', '', name)
        name = re.sub('\\\\ ', '', name)
        name = re.sub('\(\)', '', name)
        if((name[0] == 'A' or name[0] == 'B' or name[0] == 'C') and name[1] == ' '):
            name = name[2:]
        name_split = name.strip().lower().split()
        names.extend(name_split)
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        i = 0
        for paragraph in paragraphs:
            if i < 16:
                i+=1
                continue
            docs.append(paragraph)
with open("docs", "w", encoding='utf-8') as file:
   for doc in docs:
      file.write(doc)
print(names)
