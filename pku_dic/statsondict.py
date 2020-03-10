import codecs

stat={}
with codecs.open("pku_training_words.utf8", encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        stat[len(line)] = stat.get(len(line), 0) + 1
print(stat)
print("FIN")