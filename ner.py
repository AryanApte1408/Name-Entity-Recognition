import nltk
import spacy
from tabulate import tabulate

sentence = "Shah had managed to trace and interview local Taiwanese who knew of the mishap. Most told him only what they had heard of or read in a local Japanese newspaper Taiwan Nichi Nichi Shimbum. But some of them had a direct knowledge of the Indian leader's death and the disposal of what was said to be his body. Nurse Tsan Pi Sha claimed Bose had died in her presence, and Chu Tsang said he cremated his body. Chu Tsang's body was creamated on 9th october towards north-west of the castle at 9am."

print("-----------------Using NLTK----------------")
print()
data=[]
for sent in nltk.sent_tokenize(sentence):
  for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
     if hasattr(chunk, 'label'):
        data.append([chunk.label(), ' '.join(c[0] for c in chunk)])
print(tabulate(data,headers=['Label','Name Entity'],tablefmt='grid'))

print()
print("-----------------Using SpaCy----------------")
print()
data=[]
ner = spacy.load('en_core_web_sm')
doc = ner(sentence)

for ent in doc.ents:
    data.append([ent.label_,ent.text])
print(tabulate(data,headers=['Label','Name Entity'],tablefmt='grid'))

