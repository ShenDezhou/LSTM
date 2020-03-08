import codecs
import bz2
import numpy

chars = []
with codecs.open('pku_bigram_t1.utf8', 'r', encoding='utf8') as f:
    lines = f.readlines()
    for line in lines:
        for w in line:
            if w == '\n':
                continue
            else:
                chars.append(w)
print(len(chars))
rxdict = dict(zip(chars, range(1, 1 + len(chars))))

#for fast checking element in set
schars = set(chars)

bz_file = bz2.BZ2File("../model/zhwiki_20180420_100d.txt.bz2")
words, dims = bz_file.readline().strip().split(maxsplit=1)
print(words, dims)
embedding_matrix = numpy.zeros((len(chars)+1, int(dims)))

chars.append("。")

lines = bz_file.readlines()
counter = 0
lenstats = {}
for line in lines:
    line = line.strip()
    word, coefs = line.split(maxsplit=1)
    word = word.decode(encoding="utf-8")
    lenstats[len(word)] =lenstats.get(len(word), 0) + 1
    if word in schars:
        embedding_matrix[rxdict[word]] = numpy.fromstring(coefs, 'f', sep=' ')
    if counter % 10000 == 0 and counter!=0:
        print(".")
    counter += 1

print(lenstats)
print(embedding_matrix.shape)
# 4698
# 4529
#print(embedding_matrix[rxdict['。']])
zeroind = numpy.where(~embedding_matrix.any(axis=1))[0]
print(zeroind)

embedding_matrix[zeroind] = embedding_matrix[rxdict['。']]
numpy.save("../model/zhwiki_biembedding_t1.npy", embedding_matrix)

zeroind = numpy.where(~embedding_matrix.any(axis=1))[0]
print(zeroind)