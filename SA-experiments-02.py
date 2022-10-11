import sys
import pandas as pd
from experiment_tools_SA import SentimentAnalysis
from gensim.models import Word2Vec


if __name__== "__main__":
        df = pd.read_pickle('corpora/wine.pickle')
        w2v_model = Word2Vec.load("models/wine_w2v_100.model")
        hpd = {'emb_model': w2v_model}
        sa = SentimentAnalysis(hyper_params_dict=hpd,
                                df=df,
                                text_col_name='clean text',
                                label_col_name='Sentiment',
                                )
        betas1 = [0.5,1,2]#np.linspace(0.1,2,5)
        betas2 = [0.5,1,2]#np.linspace(0.5,5,5)
        nums_cols = [3,5,7]
        alphas = [0.5,0.75,0.9]#np.linspace(0.25,0.98,5)
        param_dict = {
                        'beta1':betas1,
                        'beta2':betas2,
                        'n_cols':nums_cols,
                        'alpha':alphas
                        }
        default_params_dict = {
                'beta1': 1,
                'beta2': 1,
                'n_cols': 5,
                'alpha': 0.5,
                'top_n': 50
                }
        results = sa.grid_search(param_dict=param_dict,
                                        default_params_dict=default_params_dict)
        print(f"Grid search: {results}")
        print(sa.best_accuracy,sa.best_loss)
        sys.exit()