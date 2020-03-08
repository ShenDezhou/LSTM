import codecs

from sklearn_crfsuite import metrics

MODE = 22

GOLD = '../plain/pku_test_states.txt'

if MODE == 1:
    TEST = 'pku_test_jieba_states.txt'

if MODE == 2:
    TEST = 'pku_test_ik_states.txt'

if MODE == 3:
    TEST = 'pku_test_hmmstate.utf8'

if MODE == 4:
    TEST = 'pku_test_crf_states.txt'

if MODE == 5:
    TEST = 'pku_test_hanlpsmart_states.txt'

if MODE == 6:
    TEST = 'pku_test_hanlpcrf_states.txt'

if MODE == 7:
    TEST = 'pku_test_hanlpper_states.txt'

if MODE == 8:
    TEST = 'pku_test_hanlpnlp_states.txt'

if MODE == 9:
    TEST = 'pku_test_lstm_states.txt'

if MODE == 10:
    TEST = 'pku_test_lstmbn_states.txt'

if MODE == 11:
    TEST = 'pku_test_dropout-bilstm-dropout-bn_states.txt'

if MODE == 12:
    TEST = 'pku_test_lstm_dnnbn_states.txt'

if MODE == 13:
    TEST = 'pku_test_lstmmultiple_states.txt'

if MODE == 14:
    TEST = 'pku_test_lstmow_dnn_states.txt'

if MODE == 15:
    TEST = 'pku_test_lstm_dnnbncrf_states.txt'

if MODE== 16:
    TEST = 'pku_test_hidim_bilstm_bn_states.txt'

if MODE== 17:
    TEST = 'pku_test_pretrained-bilstm-bn_states.txt'

if MODE== 18:
    TEST = 'pku_test_pretrained_hidim_bilstm_bn_states.txt'

if MODE== 19:
    TEST = 'pku_test_pretrained-ultradim-dropout-bilstm-bn_states.txt'

if MODE==20:
    TEST='pku_test_pretrained-ultradim-wide-dropout-bilstm-bn_states.txt'

if MODE==21:
    TEST='pku_test_pretrained-ultradim-wide-dropout-bilstm-bn_states.txt'

if MODE==22:
    TEST='pku_test_pretrained-ultradim-wide-dropout-bilstm-bn-crf_states.txt'

if MODE==23:
    TEST='pku_test_pretrained-extradim-wide-dropout-bilstm-bn_states.txt'

if MODE==24:
    TEST='pku_test_pretrained-extradim-wide-dropout-bilstm-bn-l1l2_states.txt'

if MODE==25:
    TEST='pku_test_pretrained_bigram_bilstm_bn_states.txt'

if MODE==26:
    TEST='pku_test_pretrained-extradim-wide-dropout-bilstm-bn-t1_states.txt'

with codecs.open(TEST, 'r', encoding='utf8') as fj:
    with codecs.open(GOLD, 'r', encoding='utf8') as fg:
        jstates = fj.readlines()
        states = fg.readlines()
        y = []
        for state in states:
            state = state.strip()
            y.append(list(state))
        yp = []
        for jstate in jstates:
            jstate = jstate.strip()
            yp.append(list(jstate))
        for i in range(len(y)):
            assert len(yp[i]) == len(y[i])
        m = metrics.flat_classification_report(
            y, yp, labels=list("BMES"), digits=4
        )
        print(m)
