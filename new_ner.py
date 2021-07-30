from numpy.lib.function_base import disp
import spacy
from spacy import displacy
path = "audax_ner/"
nlp = spacy.load(path+'lol')
thing = nlp("DIER shoes for men")
for etn in thing.ents:
    print(etn.text)
    print(etn.label_)
colors = {"BRAND":"#3dff74"}
options = {"ents": ["BRAND"], "colors": colors}
displacy.serve(thing,style="ent",options=options)
