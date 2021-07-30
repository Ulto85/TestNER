import pandas as pd
import spacy
import random
from spacy.training import Example

from spacy.util import minibatch, compounding 
nlp=spacy.load('en_core_web_sm')
#nlp = spacy.blank('en')
#nlp.add_pipe('ner')
print(nlp.pipe_names)
ner = nlp.get_pipe('ner')
path = 'audax_ner/'
df = pd.read_csv(path+'shoes.csv')
thing = ['ETC is a big brand','I recently ordered from ETC','They have a sale on ETC']
TRAIN_DATA = []

def characters(words, sentence):
    try:
        return (sentence.index(words), sentence.index(words)+len(words))
    except Exception as e:
        print(words)
        print(sentence)
        input()
for item in df["brand"]:
    sentence = random.choice(thing).replace('ETC',item)
    start, end = characters(item,sentence)
    data = (sentence, {"entities": [(start, end, "BRAND")]})
    TRAIN_DATA.append(data)
print(TRAIN_DATA)
pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
for _, info  in TRAIN_DATA:
    for ent in info.get('entities'):
        ner.add_label(ent[2])
with nlp.disable_pipes(*unaffected_pipes):
    for x in range(30):
        loss = {}
        random.shuffle(TRAIN_DATA)
        batches = minibatch(TRAIN_DATA,compounding(4.0,32.0,1.001))
        for batch in batches:
            texts, annotations = zip(*batch)
            example = []

            for i in range(len(texts)):
                doc = nlp.make_doc(texts[i])
                example.append(Example.from_dict(doc,annotations[i]))
            nlp.update(example, drop=0.5,losses=loss)
            print(loss)
        
nlp.to_disk(path+'lol')
