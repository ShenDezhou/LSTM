import glob
import os
import re

short=[
'B',
'E',
'F',
'PU',
'RU',
'FB',
'Bi',
'M',
'T',
'TD',
'A',
'RCT',
'CT',
'Ac',
'D',
'BD',
'Pd',
'C',
'Bn',
'De',
'CRF',
]
long=[
'batchsize',
'epochs',
'feature-dim',
'pretrained-unigram',
'rand-unigram',
'freqfilt-bigram',
'pretrained-bigram',
'maxmerge-bigram',
'pretrained-trigram',
'tri-dropout',
'add-ngrams',
'regularized-chartype',
'chartype',
'add-chartype',
'pre-dropout',
'bidirection',
'post-dropout',
'CuDNNLSTM',
'BatchNormalization',
'dense',
'CRF',
]

interpretdict = dict(zip(short,long))
print(interpretdict)

ROOT = r'C:\Users\Administrator\PycharmProjects\LSTM\\'

def interp(base):
    base = base.split('.')[0]
    longinterp = ""
    segs = base.split("-")
    # print(len(segs))
    for seg in segs:
        flag = re.match("([A-Za-z]*)(\d*)?", seg)
        groups = flag.groups()
        if len(flag.groups())>1:
            longinterp += interpretdict.get(groups[0], "")
            if interpretdict.get(groups[0], ""):
                if groups[1]:
                    longinterp += "=" +groups[1]
        else:
            longinterp += interpretdict.get(seg, "")
        longinterp += '-'
    longinterp = longinterp.strip('-')
    # print(base, longinterp)
    return base, longinterp


files = [f for f in glob.glob(ROOT + "*.py", recursive=True)]
for f in files:
    base = os.path.basename(f)
    shortlong = interp(base)
    if shortlong[1]:
        print(shortlong)

files = [f for f in glob.glob(ROOT + "keras/*.h5", recursive=True)]
for f in files:
    base = os.path.basename(f)
    shortlong = interp(base)
    if shortlong[1]:
        print(shortlong)

files = [f for f in glob.glob(ROOT + "baseline/*.txt", recursive=True)]
for f in files:
    base = os.path.basename(f)
    shortlong = interp(base)
    if shortlong[1]:
        print(shortlong)

