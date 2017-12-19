# coding=utf-8
import os
import numpy as np
import pandas as pd
from gensim.utils import simple_preprocess
from gensim.models.doc2vec import Doc2Vec, LabeledSentence

from time import time

start = time()


path = './wikipedia/plaintext_articles/'
fnames = os.listdir(path)

sentences = []

print( 'loading...' )
for fname in fnames:
    with open(path + fname, 'r') as f:
        doc = f.read()
    sentence = LabeledSentence(words=doc, tags=[fname])
    sentences.append(sentence)

d_vec = 400
n_doc = len(fnames)

# model = Doc2Vec(sentences, size=d_vec, alpha=0.0015, sample=1e-4, min_count=1, workers=4)
model = Doc2Vec.load('wiki.model')

print( 'training...' )
# model.train(sentences, total_examples=model.corpus_count, epochs=30)
# model.save('wiki.model')

vecs = np.empty(shape=[n_doc, d_vec], dtype=float)
vecs_df = pd.DataFrame(fnames, columns=['docname'])

for i, fname in enumerate(fnames):
    vecs[i, :] = model.docvecs[fname]

vecs_df = pd.concat([vecs_df, pd.DataFrame(vecs, columns=range(d_vec))], axis=1)
vecs_df.to_csv('./wikivec400.csv', index=False)
# print( vecs_df )

# print( model.docvecs )


print( time() - start, ': sec' )
