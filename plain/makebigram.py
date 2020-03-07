import codecs

with codecs.open('pku_training.utf8', 'r', encoding='utf8') as fa:
    with codecs.open('../pku_dic/pku_bigram.utf8', 'w', encoding='utf8') as fb:
        lines = fa.readlines()
        bigrams = []
        for line in lines:
            line = line.replace(" ", "").strip()
            chars = list(line)
            if len(chars) < 2:
                continue
            for i in range(len(chars)-1):
                bigrams.append(chars[i]+chars[i+1]+"\n")
        bigrams = list(set(bigrams))
        bigrams.sort()
        fb.writelines(bigrams)

with codecs.open('pku_training.utf8', 'r', encoding='utf8') as fa:
    with codecs.open('../pku_dic/pku_trigram.utf8', 'w', encoding='utf8') as fb:
        lines = fa.readlines()
        trigrams = []
        for line in lines:
            line = line.replace(" ", "").strip()
            chars = list(line)
            if len(chars) < 3:
                continue
            for i in range(len(chars) - 2):
                trigrams.append(chars[i] + chars[i + 1]+ chars[i + 2] + "\n")
        trigrams = list(set(trigrams))
        trigrams.sort()
        fb.writelines(trigrams)
        print("FIN")
