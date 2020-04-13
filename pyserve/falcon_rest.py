import logging

import falcon
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


l.info("create pub-bilstm-bn:")
bilstm = PUB_BiLSTM_BN()
l.info("load keras model:")
bilstm.loadKeras()
l.info("keras model loaded.")


class SegResource:

    def on_get(self, req, resp):
        """Handles GET requests"""
        resp.set_header('Access-Control-Allow-Origin', '*')
        resp.set_header('Access-Control-Allow-Methods', '*')
        resp.set_header('Access-Control-Allow-Headers', '*')
        sentence = req.get_param('q', True)
        token = req.get_param('category', False, default='accuse')
        
        sample = []
        sample_labels = []
        sample.append(sentence)

        # vectorizer = vectorizers[token]
        # features = vectorizer.transform(
        #         sample
        #         )
        # features_nd = features.toarray()
        # print("1.0 input shape", features_nd.shape)
        #
        # sample_pred = decisiontrees[token].predict(features_nd)
        # print("3:", token, "expected:", sample_labels, "actual:", sample_pred)
        print('sentence:', sentence)
        words = bilstm.cut([sentence])
        print("seg result:", words)
        print("ALL-DONE")
        resp.media = {"words":words}

    def on_post(self, req, resp):
        """Handles GET requests"""
        resp.set_header('Access-Control-Allow-Origin', '*')
        resp.set_header('Access-Control-Allow-Methods', '*')
        resp.set_header('Access-Control-Allow-Headers', '*')
        case = req.get_param('q', True)
        token = req.get_param('category', False, default='accuse')
        
        sample = []
        sample_labels = []
        sample.append(case)
        # vectorizer = vectorizers[token]
        # features = vectorizer.transform(
        #         sample
        #         )
        # features_nd = features.toarray()
        # print("1.0 input shape", features_nd.shape)
        #
        # sample_pred = decisiontrees[token].predict(features_nd)
        # print("3:", token, "expected:", sample_labels, "actual:", sample_pred)
        
        print("ALL-DONE")
        # resp.media = {"category":token, "caseid":sample_pred[0]}


api = falcon.API(middleware=[])
api.req_options.auto_parse_form_urlencoded = True
api.add_route('/segment', SegResource())
waitress.serve(api, port=8080, url_scheme='http')
