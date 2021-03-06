# spatialdropout, dropout and batchnormalization makes it SOTA.
# lstm-dnn-baseline
import codecs
import re
import string

import numpy
from keras import regularizers
from keras.layers import Dense, Embedding, LSTM,CuDNNLSTM, SpatialDropout1D, Input, Bidirectional,Dropout, BatchNormalization
from keras.models import Model
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

#               precision    recall  f1-score   support
#
#            B     0.9204    0.9280    0.9242     56882
#            M     0.6868    0.7491    0.7166     11479
#            E     0.9234    0.9284    0.9259     56882
#            S     0.9274    0.8917    0.9092     47490
#
#    micro avg     0.9063    0.9063    0.9063    172733
#    macro avg     0.8645    0.8743    0.8690    172733
# weighted avg     0.9078    0.9063    0.9068    172733

dicts = []
unidicts = []
predicts = []
sufdicts = []
longdicts = []
puncdicts = []
digitsdicts = []
chidigitsdicts = []
letterdicts = []
otherdicts = []

Thresholds = 0.95


def getTopN(dictlist):
    adict = {}
    for w in dictlist:
        adict[w] = adict.get(w, 0) + 1
    topN = max(adict.values())
    alist = [k for k, v in adict.items() if v >= topN * Thresholds]
    return alist


with codecs.open('pku_dic/pku_training_words.utf8', 'r', encoding='utf8') as fa:
    with codecs.open('pku_dic/pku_test_words.utf8', 'r', encoding='utf8') as fb:
        with codecs.open('pku_dic/contract_words.utf8', 'r', encoding='utf8') as fc:
            lines = fa.readlines()
            lines.extend(fb.readlines())
            lines.extend(fc.readlines())
            lines = [line.strip() for line in lines]
            dicts.extend(lines)
            # uni, pre, suf, long 这四个判断应该依赖外部词典，置信区间为95%，目前没有外部词典，就先用训练集词典来替代
            unidicts.extend([line for line in lines if len(line) == 1 and re.search(u'[\u4e00-\u9fff]', line)])
            predicts.extend([line[0] for line in lines if len(line) > 1 and re.search(u'[\u4e00-\u9fff]', line)])
            predicts = getTopN(predicts)
            sufdicts.extend([line[-1] for line in lines if len(line) > 1 and re.search(u'[\u4e00-\u9fff]', line)])
            sufdicts = getTopN(sufdicts)
            longdicts.extend([line for line in lines if len(line) > 3 and re.search(u'[\u4e00-\u9fff]', line)])
            puncdicts.extend(string.punctuation)
            puncdicts.extend(list("！？。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰–‘’‛“”„‟…‧﹏"))
            digitsdicts.extend(string.digits)
            chidigitsdicts.extend(list("零一二三四五六七八九十百千万亿兆〇零壹贰叁肆伍陆柒捌玖拾佰仟萬億兆"))
            letterdicts.extend(string.ascii_letters)

            somedicts = []
            somedicts.extend(unidicts)
            somedicts.extend(predicts)
            somedicts.extend(sufdicts)
            somedicts.extend(longdicts)
            somedicts.extend(puncdicts)
            somedicts.extend(digitsdicts)
            somedicts.extend(chidigitsdicts)
            somedicts.extend(letterdicts)
            otherdicts.extend(set(dicts) - set(somedicts))

chars = []

with codecs.open('pku_dic/pku_dict.utf8', 'r', encoding='utf8') as f:
    # with codecs.open('pku_diccontract_dict.utf8', 'r', encoding='utf8') as fc:
    lines = f.readlines()
    # lines.extend(fc.readlines())
    for line in lines:
        for w in line:
            if w == '\n':
                continue
            else:
                chars.append(w)
print(len(chars))

rxdict = dict(zip(chars, range(1, 1 + len(chars))))
rxdict['\n'] =0

rydict = dict(zip(list("BMES"), range(len("BMES"))))


def getNgram(sentence, i):
    ngrams = []
    ch = sentence[i]
    ngrams.append(rxdict[ch])
    return ngrams


def getFeaturesDict(sentence, i):
    features = []
    features.extend(getNgram(sentence, i))
    assert len(features) == 1
    # featuresdic = dict([(str(j), features[j]) for j in range(len(features))])
    # return featuresdic
    return features


batch_size = 64
maxlen = 1019
nFeatures = 1
word_size = 100
Hidden = 150
Regularization = 1e-4
Dropoutrate = 0.2
learningrate = 0.2
Marginlossdiscount = 0.2
nState = 4
EPOCHS = 30

loss = "squared_hinge"
optimizer = "nadam"
sequence = Input(shape=(maxlen,))
embedded = Embedding(len(chars) + 1, word_size, input_length=maxlen, mask_zero=False)(sequence)

dropout = SpatialDropout1D(rate=Dropoutrate)(embedded)
blstm = Bidirectional(CuDNNLSTM(Hidden, return_sequences=True), merge_mode='sum')(dropout)
dense = Dense(nState, activation='softmax', kernel_regularizer=regularizers.l2(Regularization))(blstm)
model = Model(input=sequence, output=dense)
model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])
model.summary()

MODE = 2

if MODE == 1:
    with codecs.open('plain/pku_training.utf8', 'r', encoding='utf8') as ft:
        with codecs.open('plain/pku_train_states.txt', 'r', encoding='utf8') as fs:
            xlines = ft.readlines()
            ylines = fs.readlines()
            X = []
            y = []

            print('process X list.')
            counter = 0
            for line in xlines:
                line = line.replace(" ", "").strip()
                line = '\n' * (maxlen - len(line)) + line
                # X.append([getFeaturesDict(line, i) for i in range(len(line))])
                X.append([rxdict.get(e, 0) for e in list(line)])
                counter += 1
                if counter % 10000 == 0 and counter != 0:
                    print('.')
            print(len(X))
            X = numpy.array(X)
            print(len(X), X.shape)

            print('process y list.')
            for line in ylines:
                line = line.strip()
                line = 'S' * (maxlen - len(line)) + line
                line = [rydict[s] for s in line]
                sline = numpy.zeros((len(line), len("BMES")), dtype=int)
                for g in range(len(line)):
                    sline[g, line[g]] = 1
                y.append(sline)
            print(len(y))
            y = numpy.array(y)
            print(len(y), y.shape)

            history = model.fit(X, y, batch_size=batch_size, nb_epoch=EPOCHS, verbose=1)

            model.save("keras/B64-E30-F1-De.h5")
            print('FIN')

if MODE == 2:
    STATES = list("BMES")
    with codecs.open('plain/pku_test.utf8', 'r', encoding='utf8') as ft:
        with codecs.open('baseline/pku_test_B64-E30-F1-De_states.txt', 'w', encoding='utf8') as fl:
            model = load_model("keras/B64-E30-F1-De.h5")
            model.summary()

            xlines = ft.readlines()
            X = []
            print('process X list.')
            counter = 0
            for line in xlines:
                line = line.replace(" ", "").strip()
                # X.append([getFeaturesDict(line, i) for i in range(len(line))])
                X.append([rxdict.get(e, 0) for e in list(line)])
                counter += 1
                if counter % 1000 == 0 and counter != 0:
                    print('.')
            print(len(X))
            X = pad_sequences(X, maxlen=maxlen, padding='pre', value=0)
            print(len(X), X.shape)
            yp = model.predict(X)
            print(yp.shape)
            for i in range(yp.shape[0]):
                sl = yp[i]
                lens = len(xlines[i].strip())
                for s in sl[-lens:]:
                    i = numpy.argmax(s)
                    fl.write(STATES[i])
                fl.write('\n')
            print('FIN')
            # for sl in yp:
            #     for s in sl:
            #         i = numpy.argmax(s)
            #         fl.write(STATES[i])
            #     fl.write('\n')
            # print('FIN')
