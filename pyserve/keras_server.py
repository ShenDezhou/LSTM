import logging
import falcon
from falcon_cors import CORS
import json
import waitress
from PUB_BiLSTM_BN import PUB_BiLSTM_BN


logging.basicConfig(level=logging.INFO, format='%(asctime)-18s %(message)s')
l = logging.getLogger()
cors_allow_all = CORS(allow_all_origins=True,
                      allow_origins_list=['http://localhost:8081'],
                      allow_all_headers=True,
                      allow_all_methods=True,
                      allow_credentials_all_origins=True
                      )

class SegResource:


    bilstm=None


    def __init__(self):
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
        l.info('sentence:', sentence)
        words = self.bilstm.cut([sentence])
        l.info("seg result:", words)
        print("ALL-DONE")
        resp.media = {"words":words}


    def on_post(self, req, resp):
        """Handles POST requests"""
        resp.set_header('Access-Control-Allow-Origin', 'http://localhost:8081')
        resp.set_header('Access-Control-Allow-Methods', '*')
        resp.set_header('Access-Control-Allow-Headers', '*')
        resp.set_header('Access-Control-Allow-Credentials', 'true')
        resp.set_header("Cache-Control", "no-cache")
        data = req.stream.read(req.content_length)
        reqdata = json.loads(data, encoding='utf-8')
        print('sentence:', reqdata['sents'])
        sentences = reqdata['sents']
        sentences = [s.strip() for s in sentences if len(s.strip())>0]
        if not isinstance(sentences, list):
            sentences = [sentences]
        segsents = self.bilstm.cut(sentences)
        print("seg result:", segsents)
        resp.media = {'data':{"seg":segsents}}


api = falcon.API(middleware=[cors_allow_all.middleware])
api.req_options.auto_parse_form_urlencoded = True
api.add_route('/segment', SegResource())
waitress.serve(api, port=8080, url_scheme='http')
