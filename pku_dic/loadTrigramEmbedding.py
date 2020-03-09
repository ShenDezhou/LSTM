import codecs
import bz2
import numpy
from scipy.sparse import csr_matrix, save_npz
#217049
#209714

#13902 trigrams
init='。'
chars = []
with codecs.open('pku_trigram.utf8', 'r', encoding='utf8') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        if len(line) > 0:
            chars.append(line)
print(len(chars))
rxdict = dict(zip(chars, range(1, 1 + len(chars))))

#for fast checking element in set
schars = set(chars)

bz_file = bz2.BZ2File("../model/zhwiki_20180420_100d.txt.bz2")
words, dims = bz_file.readline().strip().split(maxsplit=1)
print(words, dims)
embedding_matrix = numpy.zeros((len(chars)+1, int(dims)))

lines = bz_file.readlines()
counter = 0
lenstats = {}
for line in lines:
    line = line.strip()
    word, coefs = line.split(maxsplit=1)
    word = word.decode(encoding="utf-8")
    lenstats[len(word)] =lenstats.get(len(word), 0) + 1
    if word==init:
        initvector = numpy.fromstring(coefs, 'f', sep=' ')
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
print(lenstats)
print(embedding_matrix.shape)

zeroind = numpy.where(~embedding_matrix.any(axis=1))[0]
print(zeroind)

embedding_matrix[zeroind] = initvector

sparsem = csr_matrix(embedding_matrix)
save_npz("../model/zhwiki_triembedding.npz",matrix=sparsem)
# numpy.save("../model/zhwiki_triembedding.npy", embedding_matrix)

zeroind = numpy.where(~embedding_matrix.any(axis=1))[0]
print(zeroind)
print("FIN")