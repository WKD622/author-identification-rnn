import spacy

nlp = spacy.load('en_core_web_md')
doc = nlp(u'Tesla is about to buy new factory in U.S for $6 million.')

for token in doc.ents:
    print(f'{token}')
