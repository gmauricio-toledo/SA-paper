import pandas as pd
from experiment_tools_SA import SentimentAnalysis
from gensim.models import Word2Vec
import sys


if __name__== "__main__":
    df = pd.read_pickle('corpora/wine.pickle')
    w2v_model = Word2Vec.load("models/wine_w2v_100.model")
    hpd = {'emb_model': w2v_model}
    sa = SentimentAnalysis(hyper_params_dict=hpd,
                        df=df,
                        text_col_name='clean text',
                        label_col_name='Sentiment',
                        )
    comb_dict = {
            'beta1': 1,
            'beta2': 1,
            'n_cols': 5,
            'alpha': 0.9,
            'top_n': 50
            }
    results = sa.run(combination_dict=comb_dict)
    print(f"Experimento con {comb_dict}: {results}")
    sys.exit()