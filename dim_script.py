import numpy as np
import pandas as pd

import torch
import time
import torch.nn as nn

from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, RobertaModel

from scipy.spatial.distance import cdist
from skdim.id import MLE

from tqdm import tqdm


from scipy.spatial.distance import cdist
from threading import Thread

MINIMAL_CLOUD = 47

def prim_tree(adj_matrix, alpha=1.0):
    infty = np.max(adj_matrix) + 10

    dst = np.ones(adj_matrix.shape[0]) * infty
    visited = np.zeros(adj_matrix.shape[0], dtype=bool)
    ancestor = -np.ones(adj_matrix.shape[0], dtype=int)

    v, s = 0, 0.0
    for i in range(adj_matrix.shape[0] - 1):
        visited[v] = 1
        ancestor[dst > adj_matrix[v]] = v
        dst = np.minimum(dst, adj_matrix[v])
        dst[visited] = infty

        v = np.argmin(dst)
        s += (adj_matrix[v][ancestor[v]] ** alpha)

    return s.item()

def process_string(sss):
    return sss.replace('\n', ' ').replace('  ', ' ')

class PHD():
    def __init__(self, alpha=1.0, metric='euclidean', n_reruns=3, n_points=7, n_points_min=3):

# Initializes the instance of PH-dim computer
# Parameters:
# 	1) alpha --- real-valued parameter Alpha for computing PH-dim (see the reference paper). Alpha should be chosen lower than
# the ground-truth Intrinsic Dimensionality; however, Alpha=1.0 works just fine for our kind of data.
# 	2) metric --- String or Callable, distance function for the metric space (see documentation for Scipy.cdist)
# 	3) n_reruns --- Number of restarts of whole calculations (each restart is made in a separate thread)
# 	4) n_points --- Number of subsamples to be drawn at each subsample
# 	5) n_points_min --- Number of subsamples to be drawn at larger subsamples (more than half of the point cloud)
        self.alpha = alpha
        self.n_reruns = n_reruns
        self.n_points = n_points
        self.n_points_min = n_points_min
        self.metric = metric
        self.is_fitted_ = False

    def _sample_W(self, W, nSamples):
        n = W.shape[0]
        random_indices = np.random.choice(n, size=nSamples, replace=False)
        return W[random_indices]

    def _calc_ph_dim_single(self, W, test_n, outp, thread_id):
        lengths = []
        for n in test_n:
            if W.shape[0] <= 2 * n:
                restarts = self.n_points_min
            else:
                restarts = self.n_points
            reruns = np.ones(restarts)
            for i in range(restarts):
                tmp = self._sample_W(W, n)
                reruns[i] = prim_tree(cdist(tmp, tmp, metric=self.metric), self.alpha)

            lengths.append(np.median(reruns))
        lengths = np.array(lengths)

        x = np.log(np.array(list(test_n)))
        y = np.log(lengths)
        N = len(x)
        outp[thread_id] = (N * (x * y).sum() - x.sum() * y.sum()) / (N * (x ** 2).sum() - x.sum() ** 2)

    def fit_transform(self, X, y=None, min_points=50, max_points=512, point_jump=40):

# Computing the PH-dim
# Parameters:
# 	1) X --- point cloud of shape (n_points, n_features),
# 	2) y --- fictional parameter to fit with Sklearn interface
# 	3) min_points --- size of minimal subsample to be drawn
# 	4) max_points --- size of maximal subsample to be drawn
# 	5) point_jump --- step between subsamples

        ms = np.zeros(self.n_reruns)
        test_n = range(min_points, max_points, point_jump)
        threads = []

        for i in range(self.n_reruns):
            threads.append(Thread(target=self._calc_ph_dim_single, args=[X, test_n, ms, i]))
            threads[-1].start()

        for i in range(self.n_reruns):
            threads[i].join()

        m = np.mean(ms)
        return 1 / (1 - m)

from transformers import RobertaTokenizer, RobertaModel

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base')

"""
Our method (PHD) is stochastic, here are some magic constants for it. They are chosen specifically for text data. If you plan to use this code for something different, consider testing other values.

MIN_SUBSAMPLE       --- the size of the minimal subsample to be drawn in procedure. Lesser values yields less statisitcally stable predictions.
INTERMEDIATE_POINTS --- number of sumsamples to be drawn. The more this number is, the more stable dimension estimation for single text is; however,  the computational time is higher, too. 7 is, empirically, the best trade-off.
"""
MIN_SUBSAMPLE = 40
INTERMEDIATE_POINTS = 7

'''
Auxillary function. Clear text from linebreaks and odd whitespaces, because they seem to interfer with LM quite a lot.
Replace with a more sophisticated cleaner, if needed.
'''

def preprocess_text(text):
    return text.replace('\n', ' ').replace('  ', ' ')

'''
Get PHD for one text
Parameters:
        text  --- text
        solver --- PHD computator

Returns:
    real number or NumPy.nan  --- Intrinsic dimension value of the text in the input data
                                                    estimated by Persistence Homology Dimension method.'''
def get_phd_single(text, solver):
    inputs = tokenizer(preprocess_text(text), truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        outp = model(**inputs)

    # We omit the first and last tokens (<CLS> and <SEP> because they do not directly correspond to any part of the)
    mx_points = inputs['input_ids'].shape[1] - 2


    mn_points = MIN_SUBSAMPLE
    step = ( mx_points - mn_points ) // INTERMEDIATE_POINTS
    if step == 0: step = 1

    return solver.fit_transform(outp[0][0].numpy()[1:-1],  min_points=mn_points, max_points=mx_points - step, \
                                point_jump=step)

'''
Get PHD for all texts in df[key] Pandas DataSeries (PHD method)
Parameters:
        df  --- Pandas DataFrame
        key --- Name of the column
        is_list --- Check if the elements of the df[key] are lists (appears in some data)

        alpha --- Parameter alpha for PHD computattion

Returns:
    numpy.array of shape (number_of_texts, 1) --- Intrinsic dimension values for all texts in the input data
                                                    estimated by Persistence Homology Dimension method.
'''

def get_phd(df, train_idx, split, key='text', is_list=False, alpha=1.0):
    dims = []
    PHD_solver = PHD(alpha=alpha, metric='euclidean', n_points=9)
    # for s in tqdm(df[key]):
    #     if is_list:
    #         text = s[0]
    #     else:
    #         text = s
    i = 0
    for s in tqdm(df):
        if i >= train_idx: break
        if s['split'] != split: continue
        if is_list:
            text = s[key][0]
        else: text = s[key]
        dims.append(get_phd_single(text, PHD_solver))
        i += 1

    return np.array(dims).reshape(-1, 1)

'''
Get MLE for one text
Parameters:
        text  --- text
        solver --- MLE computator

Returns:
    real number or NumPy.nan  --- Intrinsic dimension value of the text in the input data
                                                    estimated by Maximum Likelihood Estimation method.'''
def get_mle_single(text, solver):
    inputs = tokenizer(preprocess_text(text), truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        outp = model(**inputs)

    return solver.fit_transform(outp[0][0].numpy()[1:-1])

'''
Get PHD for all texts in df[key] Pandas DataSeries (PHD method)
Parameters:
        df  --- Pandas DataFrame
        key --- Name of the column
        is_list --- Check if the elements of the df[key] are lists (appears in some data)

Returns:
    numpy.array of shape (number_of_texts, 1) --- Intrinsic dimension values for all texts in the input data
                                                    estimated by Maximum Likelihood Estimation method.
'''

def get_mle(df, train_idx, split, key='text', is_list=False):
    dims = []
    MLE_solver = MLE()
    # for s in tqdm(df[key]):
    #     if is_list:
    #         text = s[0]
    #     else:
    #         text = s
    #     dims.append(get_mle_single(text, MLE_solver))
    i = 0
    for s in tqdm(df):
        if i >= train_idx: break
        if s['split'] != split: continue
        if is_list:
            text = s[key][0]
        else: text = s[key]
        dims.append(get_mle_single(text, MLE_solver))
        i += 1

    return np.array(dims).reshape(-1, 1)

def create_article(title, link, year, texts):
    return {
        "title": title,
        "link": link,
        "year": year,
        "texts": texts
    }
def create_pair(generated, real):
    return {
        "generated": generated,
        "real": real
    }
def create_trio_dim(text, dim_PHD, dim_MLE):
    return {
        "text": text,
        "dim_PHdim": dim_PHD,
        "dim_MLE": dim_MLE
    }

import json
from tqdm import tqdm
from progress.bar import IncrementalBar

with open('gen_human_papers.json', 'r', encoding="utf-8") as file:
    data = json.load(file)

solver = PHD(alpha=1.0, metric='euclidean', n_points=9)
MLE_solver = MLE()

articles = []

print(len(data))

import random

# bar = IncrementalBar('Countdown', max = len(data))
random.seed(42)
random.shuffle(data)

for elem in tqdm(data):
    #     bar.next()
    sets = elem['texts']
    link = elem['link']
    year = elem['year']
    title = elem['title']
    pairs = []
    times_mle = []
    times_phd = []
    for st in sets:
        text_real = st['real']
        text_gen = st['generated']

        start_time = time.perf_counter()
        dim_PHdim_real = get_phd_single(text_real, solver)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        times_phd.append(elapsed_time)

        start_time = time.perf_counter()
        dim_PHdim_gen = get_phd_single(text_gen, solver)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        times_phd.append(elapsed_time)

        start_time = time.perf_counter()
        dim_MLE_real = get_mle_single(text_real, MLE_solver)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        times_mle.append(elapsed_time)

        start_time = time.perf_counter()
        dim_MLE_gen = get_mle_single(text_gen, MLE_solver)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        times_mle.append(elapsed_time)


        trio_real = create_trio_dim(text_real, dim_PHdim_real, dim_MLE_real)
        trio_gen = create_trio_dim(text_gen, dim_PHdim_gen, dim_MLE_gen)
        pair = create_pair(trio_gen, trio_real)
        pairs.append(pair)
    article = create_article(title, link, year, pairs)
    articles.append(article)
    print("MLE time", sum(times_mle)/len(times_mle))
    print("PHD time", sum(times_phd) / len(times_phd))
    json_data = json.dumps(article, indent=4, ensure_ascii=False, separators=(',', ': '))
    with open('dims/' + str(hash(link)) + '.json', 'w', encoding='utf-8') as file:
        file.write(json_data)

json_data = json.dumps(articles, indent=4, ensure_ascii=False, separators=(',', ': '))
with open('dims_GenHuman_papers.json', 'w', encoding='utf-8') as file:
    file.write(json_data)










