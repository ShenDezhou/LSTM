# pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
import codecs
import re
import string
import pickle

import numpy
from keras import regularizers
from keras.layers import Dense, Embedding, Add, SpatialDropout1D, LSTM, Input, Bidirectional, Flatten,CuDNNLSTM, Lambda, Dropout, BatchNormalization, Average, concatenate
from keras.models import Model
from keras.models import load_model
from keras.optimizers import Adagrad
from sklearn_crfsuite import metrics
from keras.preprocessing.sequence import pad_sequences
import keras.backend as K
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_accuracy

#               precision    recall  f1-score   support
#
#            B     0.5050    0.1374    0.2160     56882
#            M     0.0000    0.0000    0.0000     11479
#            E     0.5250    0.1416    0.2231     56882
#            S     0.3170    0.9474    0.4751     47490
#
#    micro avg     0.3523    0.3523    0.3523    172733
#    macro avg     0.3368    0.3066    0.2285    172733
# weighted avg     0.4264    0.3523    0.2752    172733

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
def getCharType(ch):
    types = []

    dictofdicts = [puncdicts, digitsdicts, chidigitsdicts, letterdicts, unidicts, predicts, sufdicts]
    for i in range(len(dictofdicts)):
        if ch in dictofdicts[i]:
            types.append(i)
            break

    extradicts = [longdicts, otherdicts]
    for i in range(len(extradicts)):
        for word in extradicts[i]:
            if ch in word:
                types.append(i + len(dictofdicts))
                break
        if len(types) > 0:
            break

    if len(types) == 0:
        return str(len(dictofdicts) + len(extradicts) - 1)

    assert len(types) == 1 or len(types) == 2, "{} {} {}".format(ch, len(types), types)
    # onehot = [0] * (len(dictofdicts) + len(extradicts))
    # for i in types:
    #     onehot[i] = 1

    return str(types[0])


def safea(sentence, i):
    if i < 0:
        return '\n'
    if i >= len(sentence):
        return '\n'
    return sentence[i]


def getNgram(sentence, i):
    #5 + 4*2 + 2*3=19
    ngrams = []
    for offset in [-2, -1, 0, 1, 2]:
        ngrams.append(safea(sentence, i + offset))

    for offset in [-2, -1, 0, 1]:
        ngrams.append(safea(sentence, i + offset) + safea(sentence, i + offset + 1))

    for offset in [-1, 0]:
        ngrams.append(safea(sentence, i + offset) + safea(sentence, i + offset + 1) + safea(sentence, i + offset + 2))
    return ngrams

def getBigram(sentence, i):
    #5 + 4*2 + 2*3=19
    ngrams = []
    for offset in [0, 1, 2]:
        ngrams.append(safea(sentence, i + offset))
    return ngrams


def getBigramVector(sentence, i):
    ngrams = getBigram(sentence, i)
    ngramv = []
    for ngram in ngrams:
        for ch in ngram:
            ngramv.append(rxdict.get(ch,0))
    return ngramv


def getNgramVector(sentence, i):
    ngrams = getNgram(sentence, i)
    ngramv = []
    for ngram in ngrams:
        for word in ngram:
            for ch in word:
                ngramv.append(rxdict.get(ch,0))
    return ngramv

def getReduplication(sentence, i):
    reduplication = []
    for offset in [-2, -1]:
        if safea(sentence, i) == safea(sentence, i + offset):
            reduplication.append('1')
        else:
            reduplication.append('0')
    return reduplication

def getReduplicationVector(sentence, i):
    reduplicationv =[int(e) for e in getReduplication(sentence,i)]
    return reduplicationv

def getType(sentence, i):
    types = []
    for offset in [-1, 0, 1]:
        types.append(getCharType(safea(sentence, i + offset)))
    # types.append(getCharType(safea(sentence, i + offset - 1)) + getCharType(safea(sentence, i + offset)) + getCharType(
    #         safea(sentence, i + offset + 1)))
    return types

def getTypeVector(sentence, i):
    types = getType(sentence,i)
    types = [int(t) for t in types]
    return types

def getFeatures(sentence, i):
    features = []
    features.extend(getBigramVector(sentence, i))
    # features.extend(getReduplicationVector(sentence, i))
    # features.extend(getTypeVector(sentence, i))
    assert len(features) == 3, (len(features),features)
    return features


def getFeaturesDict(sentence, i):
    features = []
    features.extend(getNgramVector(sentence, i))
    features.extend(getReduplicationVector(sentence, i))
    features.extend(getType(sentence, i))
    assert len(features) == 24
    featuresdic = dict([(str(j), features[j]) for j in range(len(features))])
    return featuresdic

batch_size = 64
maxlen = 1019
nFeatures = 3
word_size = 100
Hidden = 150
Regularization = 1e-4
Dropoutrate = 0.2
learningrate = 0.2
Marginlossdiscount = 0.2
nState = 4
EPOCHS = 50



MODE = 3

if MODE == 1:
    with codecs.open('plain/pku_training.utf8', 'r', encoding='utf8') as ft:
        with codecs.open('plain/pku_train_states.txt', 'r', encoding='utf8') as fs:
            with codecs.open('model/pku_train_crffeatures.pkl', 'wb') as fx:
                with codecs.open('model/pku_train_crfstates.pkl', 'wb') as fy:
                    xlines = ft.readlines()
                    ylines = fs.readlines()
                    X = []
                    y = []

                    print('process X list.')
                    counter = 0
                    for line in xlines:
                        line = line.replace(" ", "").strip()
                        line = '\n' *(maxlen-len(line)) + line
                        assert len(line)==maxlen
                        X.append([getFeatures(line, i) for i in range(len(line))])
                        # X.append([rxdict.get(e, 0) for e in list(line)])
                        # break
                        counter += 1
                        if counter % 10000 == 0 and counter != 0:
                            print('.')

                    X = numpy.array(X)
                    print(len(X), X.shape)
                    # X = pad_sequences(X, maxlen=maxlen, padding='pre', value=[0]*nFeatures)
                    # print(len(X), X.shape)

                    print('process y list.')
                    for line in ylines:
                        line = line.strip()
                        line = 'S' *(maxlen-len(line)) + line
                        line = [rydict[s] for s in line]
                        sline = numpy.zeros((len(line), len("BMES")), dtype=int)
                        for g in range(len(line)):
                            sline[g, line[g]] = 1
                        y.append(sline)
                        # break
                    print(len(y))
                    # y = pad_sequences(y, maxlen=maxlen, padding='pre', value=3)
                    y = numpy.array(y)
                    print(len(y), y.shape)

                    print('validate size.')
                    for i in range(len(X)):
                        assert len(X[i]) == len(y[i])

                    print('output to file.')
                    sX = pickle.dumps(X)
                    fx.write(sX)
                    sy = pickle.dumps(y)
                    fy.write(sy)

if MODE==2:
    loss = crf_loss
    optimizer = "nadam" #Adagrad(lr=0.2) # "adagrad"
    metric= crf_accuracy
    sequence = Input(shape=(maxlen,nFeatures,))
    seqsa, seqsb, seqsc = Lambda(lambda x: [x[:,:,0],x[:,:,1],x[:,:,2]])(sequence)
    embeddeda = Embedding(len(chars) + 1, word_size, input_length=maxlen, mask_zero=False)(seqsa)
    # dropouta = SpatialDropout1D(rate=Dropoutrate)(embeddeda)
    embeddedb = Embedding(len(chars) + 1, word_size, input_length=maxlen, mask_zero=False)(seqsb)
    # dropoutb = SpatialDropout1D(rate=Dropoutrate)(embeddedb)
    embeddedc = Embedding(len(chars) + 1, word_size, input_length=maxlen, mask_zero=False)(seqsc)
    # dropoutc = SpatialDropout1D(rate=Dropoutrate)(embeddedc)

    averagea = Average()([embeddeda, embeddedb])
    averageb = Average()([embeddedc, embeddedb])

    concat = Add()([embeddeda, averagea,averageb])

    blstm = Bidirectional(CuDNNLSTM(Hidden,batch_input_shape=(maxlen,nFeatures), return_sequences=True), merge_mode='sum')(concat)
    dropout = Dropout(rate=Dropoutrate)(blstm)
    batchNorm = BatchNormalization()(dropout)
    dense = Dense(nState, activation='softmax', kernel_regularizer=regularizers.l2(Regularization))(batchNorm)
    crf = CRF(nState, activation='softmax', kernel_regularizer=regularizers.l2(Regularization))(dense)

    model = Model(input=sequence, output=crf)
    # model.compile(loss='categorical_crossentropy', optimizer=adagrad, metrics=["accuracy"])
    # optimizer = Adagrad(lr=learningrate)
    model.compile(loss=loss, optimizer=optimizer, metrics=[metric])
    model.summary()

    with codecs.open('model/pku_train_crffeatures.pkl', 'rb') as fx:
        with codecs.open('model/pku_train_crfstates.pkl', 'rb') as fy:
            with codecs.open('model/pku_train_lstmmodel.pkl', 'wb') as fm:
                bx = fx.read()
                by = fy.read()
                X = pickle.loads(bx)
                y = pickle.loads(by)
                print(X[-1])
                print(y[-1])
                for i in range(len(X)):
                    assert len(X[i]) == len(y[i])
                print('training')

                history = model.fit(X, y, batch_size=batch_size, nb_epoch=EPOCHS, verbose=1)

                print('trained')
                sm = pickle.dumps(model)
                fm.write(sm)

                # yp = model.predict(X)
                # print(yp)
                # m = metrics.flat_classification_report(
                #     y, yp, labels=list("BMES"), digits=4
                # )
                # print(m)
                model.save("keras/B64-E50-F3-A-Bn-De-CRF.h5")
                print('FIN')

if MODE == 3:
    STATES = list("BMES")
    with codecs.open('plain/pku_test.utf8', 'r', encoding='utf8') as ft:
        with codecs.open('baseline/pku_test_B64-E50-F3-A-Bn-De-CRF_states.txt', 'w', encoding='utf8') as fl:
            custom_objects = {'CRF': CRF,
                              'crf_loss': crf_loss,
                              'crf_accuracy': crf_accuracy}
            model = load_model("keras/B64-E50-F3-A-Bn-De-CRF.h5",custom_objects=custom_objects)
            model.summary()

            xlines = ft.readlines()
            X = []
            print('process X list.')
            counter = 0
            for line in xlines:
                line = line.replace(" ", "").strip()
                line = '\n' * (maxlen - len(line)) + line
                X.append([getFeatures(line, i) for i in range(len(line))])
                # X.append([rxdict.get(e, 0) for e in list(line)])
                counter += 1
                if counter % 1000 == 0 and counter != 0:
                    print('.')
            X = numpy.array(X)
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

