import codecs

from sklearn_crfsuite import metrics

MODE = 11

GOLD = '../plain/pku_test_states.txt'

if MODE == 1:
    TEST = 'pku_test_pretrained_context_unigram_states.txt'

if MODE == 2:
    TEST = 'pku_test_f1_states.txt'

if MODE == 3:
    TEST = 'pku_test_pretrained-context3-unigram-chartype_states.txt'

if MODE == 4:
    TEST = 'pku_test_B256-E30-F1_states.txt'

if MODE == 5:
    TEST = 'pku_test_B20-E30-F1_states.txt'

if MODE == 6:
    TEST = 'pku_test_B256-E30-F8-RU-A-CT-Ac_states.txt'

if MODE == 7:
    TEST = 'pku_test_B256-E30-F3-RU-Bn-De_states.txt'

if MODE == 8:
    TEST = 'pku_test_B20-E60-F1-PU-Bn-De_states.txt'

if MODE == 9:
    TEST = 'pku_test_B64-E30-F1-Bn-CRF_states.txt'

if MODE == 10:
    TEST = 'pku_test_B20-E30-F8-PU-CT-Bn-De-CRF_states.txt'

if MODE == 11:
    TEST = 'pku_test_B64-E50-F3-A-Bn-De-CRF_states.txt'

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
    TEST='pku_test_pretrained-extradim-bigram-dict-dropout-bilstm-bn_states.txt'

if MODE==24:
    TEST='pku_test_pretrained-extradim-wide-dropout-bilstm-bn-l1l2_states.txt'

if MODE==25:
    TEST='pku_test_pretrained_bigram_bilstm_bn_states.txt'

if MODE==26:
    TEST='pku_test_pretrained-extradim-wide-dropout-bilstm-bn-t1_states.txt'

if MODE==27:
    TEST='pku_test_pretrained-extradim-initial-dropout-bilstm-bn_states.txt'

if MODE==28:
    TEST='pku_test_pretrained-extradim-trigram-dropout-bilstm-bn_states.txt'

if MODE==29:
    TEST='pku_test_pretrained-elevendim-trigram-dropout-bilstm_states.txt'

if MODE==30:
    TEST='pku_test_pretrained-elevendim-trigram-dropout-bilstm-bn_states.txt'

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
