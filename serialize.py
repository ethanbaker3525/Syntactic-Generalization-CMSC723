import stanza
import nltk
import pandas as pd

def download_data():
    stanza.download("en")
    nltk.download("brown")

# download_data()

nlp = stanza.Pipeline(lang="en", processors="tokenize,pos,constituency")

sentences_to_parse = nltk.corpus.brown.sents()

parsed_trees = []

print("Parsing sentences...")
for i, tokens in enumerate(sentences_to_parse):
    raw_sentence = ' '.join(tokens) # brown is in token form, we want sentences
    doc = nlp(raw_sentence)
    tree = doc.sentences[0].constituency
    parsed_trees.append(tree)
    
    if (i + 1) % 100 == 0:
        print(f"Parsing sentence {i+1}...")

df = pd.DataFrame({
    "sentence_id": range(1, len(parsed_trees) + 1),
    "tree": [str(tree) for tree in parsed_trees]
})

df.to_csv("Data/trees/brown_trees.tsv", sep="\t", index=False)


    
    
