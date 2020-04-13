import logging

import falcon
from falcon_cors import CORS
import json
import waitress

from sklearn.feature_extraction.text import CountVectorizer
import os
import codecs
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy
from sklearn.model_selection import train_test_split
import pkuseg
import datetime
import pickle
from functools import partial
from pathlib import Path
import argparse

from PUB_BiLSTM_BN import PUB_BiLSTM_BN

class RequireJSON(object):

    def process_request(self, req, resp):
        if not req.client_accepts_json:
            raise falcon.HTTPNotAcceptable(
                'This API only supports responses encoded as JSON.',
                href='http://docs.examples.com/api/json')

        if req.method in ('POST', 'PUT'):
            if 'application/json' not in req.content_type:
                raise falcon.HTTPUnsupportedMediaType(
                    'This API only supports requests encoded as JSON.',
                    href='http://docs.examples.com/api/json')


class JSONTranslator(object):
    # NOTE: Starting with Falcon 1.3, you can simply
    # use req.media and resp.media for this instead.

    def process_request(self, req, resp):
        # req.stream corresponds to the WSGI wsgi.input environ variable,
        # and allows you to read bytes from the request body.
        #
        # See also: PEP 3333
        if req.content_length in (None, 0):
            # Nothing to do
            return

        body = req.stream.read()
        if not body:
            raise falcon.HTTPBadRequest('Empty request body',
                                        'A valid JSON document is required.')

        try:

            req.context['doc'] = json.loads(body.decode('utf-8'))

        except (ValueError, UnicodeDecodeError):
            raise falcon.HTTPError(falcon.HTTP_753,
                                   'Malformed JSON',
                                   'Could not decode the request body. The '
                                   'JSON was incorrect or not encoded as '
                                   'UTF-8.')

    def process_response(self, req, resp, resource):
        if 'result' not in resp.context:
            return

        resp.body = json.dumps(resp.context['result'])


# def cut(text, aseg):
#     return aseg.cut(text)


# vectorizers = {}
# decisiontrees = {}
# tokens = ["execution", "civil", "criminal", "administrative", "statecompensation", "accuse"]
# gapmodel = None
# for token in tokens:
#     with open("./model/" + token + "countvectorizer.pkl", "rb") as f:
#         s = f.read()
#         print("1:vectorizer", len(s))
#         vectorizers[token] = pickle.loads(s)
#     with open("./model/" + token + "decisiontree.pkl", "rb") as f:
#         s = f.read()
#         decisiontrees[token] = pickle.loads(s)
#         print("2:decisiontree", len(s))
# #     if token == "accuse":
# #         with open("./model/" + token + "labelencoder.pkl", "rb") as f:
# #             s = f.read()
# #             lbe = pickle.loads(s)
# #             print("2.1:labelencoder", len(s))
#
logging.basicConfig(level=logging.INFO, format='%(asctime)-18s %(message)s')
l = logging.getLogger()

cors_allow_all = CORS(allow_all_origins=True,
                      allow_origins_list=['http://localhost:8081'],
                      allow_all_headers=True,
                      allow_all_methods=True,
                      allow_credentials_all_origins=True
                      )


class SegResource:
    #instance variables
    bilstm=None

    def __init__(self):
        # return
        l.info("create pub-bilstm-bn:")
        self.bilstm = PUB_BiLSTM_BN()
        l.info("load keras model:")
        self.bilstm.loadKeras()
        l.info("keras model loaded.")
        segs = self.bilstm.cut(["我昨天去清华大学。", "他明天去北京大学，再后天去麻省理工大学。"])
        l.info("inference done.")
        print(segs)

    def on_get(self, req, resp):
        """Handles GET requests"""
        resp.set_header('Access-Control-Allow-Origin', 'http://localhost:8081')
        resp.set_header('Access-Control-Allow-Methods', '*')
        resp.set_header('Access-Control-Allow-Headers', '*')
        resp.set_header('Access-Control-Allow-Credentials','true')
        sentence = req.get_param('q', True)
        # token = req.get_param('category', False, default='accuse')
        
        # sample = []
        sample_labels = []
        # sample.append(sentence)

        # vectorizer = vectorizers[token]
        # features  = vectorizer.transform(
        #         sample
        #         )
        # features_nd = features.toarray()
        # print("1.0 input shape", features_nd.shape)
        #
        # sample_pred = decisiontrees[token].predict(features_nd)
        # print("3:", token, "expected:", sample_labels, "actual:", sample_pred)
        print('sentence:', sentence)
        words = self.bilstm.cut([sentence])
        print("seg result:", words)
        print("ALL-DONE")
        resp.media = {"words":words}

    # {'q':['list of sentences to be segged']}
    #
    def on_post(self, req, resp):
        """Handles GET requests"""
        resp.set_header('Access-Control-Allow-Origin', 'http://localhost:8081')
        resp.set_header('Access-Control-Allow-Methods', '*')
        resp.set_header('Access-Control-Allow-Headers', '*')
        resp.set_header('Access-Control-Allow-Credentials', 'true')
        resp.set_header("Cache-Control", "no-cache")
        # sentence = req.get_param('q', True)
        #token = req.get_param('category', False, default='accuse')
        data = req.stream.read(req.content_length)
        reqdata = json.loads(data, encoding='utf-8')

        print('sentence:', reqdata['sents'])
        sentences = reqdata['sents']
        sentences = [s.strip() for s in sentences if len(s.strip())>0]
        if not isinstance(sentences, list):
            sentences = [sentences]
        segsents = self.bilstm.cut(sentences)
        print("seg result:", segsents)
        print("ALL-DONE")
        resp.media = {'data':{"seg":segsents}}


api = falcon.API(middleware=[cors_allow_all.middleware])
api.req_options.auto_parse_form_urlencoded = True
api.add_route('/segment', SegResource())
waitress.serve(api, port=8080, url_scheme='http')
