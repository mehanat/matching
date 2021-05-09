from flask import Flask
from flask import request
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from langdetect import detect
import os
import string
import nltk
import torch
import torch.nn.functional as F
from collections import Counter
from typing import Dict, List, Tuple, Union, Callable
#import faiss


app = Flask(__name__)

ids = []
sentences = []
sentences_emb = []
embedding_data = dict()
emb_matrix = None
vocab = None
unk_words = None
tfidf = None
tfidf_vectorizer = None
knrm = None
all_tokens = []
index = ''#None
oov_val = 1
PAD = np.zeros(50)

@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/ping')
def ping():
    return {'status': 'ok'}

@app.route('/query', methods=['POST'])
def query():
    res = {
        'lang_check': [],
        'suggestions': []
    }
    if index is None:
        return {'status': 'FAISS is not initialized!'}
    else:
        queries = request.get_json()['queries']
        for query in queries:
            lang = detect(query)
            res['lang_check'].append(lang)
            if lang == 'en':
                res['suggestions'].append(get_suggestions(query))
    print('res', res)
    return res

@app.route('/update_index', methods=['POST'])
def update_index():
    print('update_index')
    documents = request.get_json()['documents']
    global sentences
    global ids
    global emb_matrix
    global vocab

    for id, sentence in documents.items():
        ids.append(id)
        sentences.append(sentence)

    matrix, vocab, unk_words = process_data(sentences)
    emb_matrix = matrix
    vocab = vocab
    print('vocab', vocab)

    sentences = np.array(sentences)
    ids = np.array(ids)
    global tfidf_vectorizer
    global tfidf
    global embeddings
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, vocabulary=vocab.keys())
    tfidf = tfidf_vectorizer.fit_transform(sentences)

    features = np.array(tfidf_vectorizer.get_feature_names())
    for sent in tfidf:
        words = features[sent.indices]
        data = sent.data
        l = len(data)
        res = np.zeros(50)
        for j in range(l):
            res += (embedding_data[words[j]] * sent.data[j])
        sentences_emb.append(res)

    global sentences_emb_np
    sentences_emb_np = np.array(sentences_emb, dtype=np.float32)
    global index
    #index = faiss.index_factory(50, "IVF32_HNSW10,Flat")
    #index.train(sentences_emb_np)
    #index.add(sentences_emb_np)
    #index = 'ss'
    global knrm
    knrm = KNRM(emb_matrix, freeze_embeddings=True)
    #knrm.load_state_dict(torch.load('C:\\Users\\User\\Downloads\\knrm'))
    #knrm.eval()
    return {'status': 'ok'}

def preproc_func(inp_str):
    inp_str = str(inp_str)
    inp_str = inp_str.strip().lower()
    for punct in string.punctuation:
        inp_str = inp_str.replace(punct, ' ')
    return nltk.word_tokenize(inp_str)

def _tokenized_text_to_index(tokenized_text, pad_to = 10):
    res = [vocab.get(i, oov_val) for i in tokenized_text]
    return res


def _convert_text_idx_to_token_idxs(sentence):
    tokenized_text = preproc_func(sentence)
    idxs = _tokenized_text_to_index(tokenized_text)
    return idxs

def get_suggestions(query):
    res = []
    query_ind = _convert_text_idx_to_token_idxs(query)
    distances, indices = [], np.array([1,2,3,4,5,6,7,8,9])#index.search(query, 10)

    suggestions = list([
        torch.tensor(_convert_text_idx_to_token_idxs(sentences[ind])).reshape(1, -1)
        for ind in indices])
    query_ind = torch.tensor(query_ind).reshape(1, -1)
    logits = list([
        knrm.predict({
            'query': query_ind,
            'document': doc_ind
        }).detach().numpy()[0][0] for doc_ind in suggestions
    ])
    sorted_perm = np.array(sorted(range(len(logits)), key=lambda k: logits[k]))
    return list([(
        int(indices[i]), sentences[indices[i]]
    ) for i in sorted_perm])

def _filter_rare_words(vocab, min_occurancies):
    out_vocab = dict()
    for word, cnt in vocab.items():
        if cnt >= min_occurancies:
            out_vocab[word] = cnt
    return out_vocab

def process_data(sentences):
    def flatten(t): return [item for sublist in t for item in sublist]
    tokens = []
    unique_texts = set(sentences)
    tokens.extend(list(map(preproc_func, unique_texts)))
    global all_tokens
    all_tokens = list(
        _filter_rare_words(Counter(flatten(tokens)), 0.01).keys()
    )

    inner_keys = ['PAD', 'OOV'] + all_tokens
    input_dim = len(inner_keys)
    out_dim = 50

    vocab = dict()
    matrix = np.empty((input_dim, out_dim))
    unk_words = []
    for idx, word in enumerate(inner_keys):
        vocab[word] = idx
        if word in embedding_data:
            matrix[idx] = embedding_data[word]
        else:
            unk_words.append(word)
            matrix[idx] = np.random.uniform(-0.2, 0.2, size=out_dim)
    matrix[0] = np.zeros_like(matrix[0])
    return matrix, vocab, unk_words

def read_glove_embeddings(file_path):
    print('read_glove_embeddings')
    embedding_data = dict({})
    with open(file_path, encoding="utf8") as f:
        for line in f:
            l = line.split()
            embedding_data[l[0]] = np.array(list(map(float, l[1:])))
    print('embedding_data size', len(embedding_data))
    return embedding_data

def get_sentence_embedding(sentence, tfidf_vectorizer, features, embeddings):
    scores = tfidf_vectorizer.transform([sentence])
    words = features[scores.indices]
    res = np.zeros(50)
    for i in range(len(scores.data)):
        res += (embeddings[words[i]] * scores.data[i])
    return res

class GaussianKernel(torch.nn.Module):
    def __init__(self, mu: float = 1., sigma: float = 1.):
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def forward(self, x):
        return torch.exp(
            -0.5 * ((x - self.mu) ** 2) / (self.sigma ** 2)
        )


class KNRM(torch.nn.Module):
    def __init__(self, embedding_matrix: np.ndarray, freeze_embeddings: bool, kernel_num: int = 21,
                 sigma: float = 0.1, exact_sigma: float = 0.001,
                 out_layers: List[int] = [10, 5]):
        super().__init__()
        self.embeddings = torch.nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix),
            freeze=freeze_embeddings,
            padding_idx=0
        )

        self.kernel_num = kernel_num
        self.sigma = sigma
        self.exact_sigma = exact_sigma
        self.out_layers = out_layers

        self.kernels = self._get_kernels_layers()

        self.mlp = self._get_mlp()

        self.out_activation = torch.nn.Sigmoid()

    def _get_kernels_layers(self) -> torch.nn.ModuleList:
        kernels = torch.nn.ModuleList()
        for i in range(self.kernel_num):
            mu = 1. / (self.kernel_num - 1) + (2. * i) / (
                self.kernel_num - 1) - 1.0
            sigma = self.sigma
            if mu > 1.0:
                sigma = self.exact_sigma
                mu = 1.0
            kernels.append(GaussianKernel(mu=mu, sigma=sigma))
        return kernels

    def _get_mlp(self) -> torch.nn.Sequential:
        out_cont = [self.kernel_num] + self.out_layers + [1]
        mlp = [
            torch.nn.Sequential(
                torch.nn.Linear(in_f, out_f),
                torch.nn.ReLU()
            )
            for in_f, out_f in zip(out_cont, out_cont[1:])
        ]
        mlp[-1] = mlp[-1][:-1]
        return torch.nn.Sequential(*mlp)

    def forward(self, input_1: Dict[str, torch.Tensor], input_2: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        logits_1 = self.predict(input_1)
        logits_2 = self.predict(input_2)

        logits_diff = logits_1 - logits_2

        out = self.out_activation(logits_diff)
        return out

    def _get_matching_matrix(self, query: torch.Tensor, doc: torch.Tensor) -> torch.FloatTensor:
        # shape = [B, L, D]
        embed_query = self.embeddings(query.long())
        # shape = [B, R, D]
        embed_doc = self.embeddings(doc.long())

        # shape = [B, L, R]
        matching_matrix = torch.einsum(
            'bld,brd->blr',
            F.normalize(embed_query, p=2, dim=-1),
            F.normalize(embed_doc, p=2, dim=-1)
        )
        return matching_matrix

    def _apply_kernels(self, matching_matrix: torch.FloatTensor) -> torch.FloatTensor:
        KM = []
        for kernel in self.kernels:
            # shape = [B]
            K = torch.log1p(kernel(matching_matrix).sum(dim=-1)).sum(dim=-1)
            KM.append(K)

        # shape = [B, K]
        kernels_out = torch.stack(KM, dim=1)
        return kernels_out

    def predict(self, inputs: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        query, doc = inputs['query'], inputs['document']
        # shape = [B, L, R]
        matching_matrix = self._get_matching_matrix(query, doc)
        # shape [B, K]
        kernels_out = self._apply_kernels(matching_matrix)
        # shape [B]
        out = self.mlp(kernels_out)
        return out

if __name__ == '__main__':
    EMB_PATH_GLOVE = os.environ.get("EMB_PATH_GLOVE")
    print(EMB_PATH_GLOVE)
    embedding_data = read_glove_embeddings(EMB_PATH_GLOVE)

    print('starting..')
    app.run()
