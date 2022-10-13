from itertools import product
from copy import Error
from logging import raiseExceptions
import numpy as np
import pandas as pd
from scoring import Scoring
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.utils import np_utils
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from SentimentKW import KW
import time


class Iterador:

    def __init__(self,kw_dict,emb_model,df,n_clases):
        '''
        El dataframe debe contener el texto limpio, como string, en una columna llamada "text". La etiqueta debe estar en 
        una columna "Sentiment", como enteros 0,1,...,n_clases-1.
        '''
        self.kw_dict = kw_dict
        self.emb_model = emb_model
        self.df = df.copy()
        self.n_clases = n_clases
        self.y = self.df['Sentiment'].values
        self.df['Normalized Sentiment'] = (1/2)*self.df['Sentiment'].values-1 # AJUSTAR PARA n_clases

    def iterar(self,alpha,beta1,beta2,n_cols,n_iter=5):
        self.n_cols = n_cols
        losses = []
        accs = []
        for k in range(n_iter):
            puntaje = Scoring(w2vmodel=self.emb_model,W0=self.kw_dict)
            puntaje.build_neighbors(alpha=alpha)
            scores_df = puntaje.transform(df=self.df,text_col="text",label_col="Normalized Label",beta1=beta1,beta2=beta2)
            kw_dict = dict(zip(scores_df['word'].values,scores_df['score'].values))
            _ = puntaje.get_words_representations(mode='mean')
            X_msjs_rep = puntaje.get_texts_representations_MAT(cols_num=n_cols)
            X_train, X_test, y_train, y_test = train_test_split(X_msjs_rep, self.y, random_state=331,train_size=0.75) 
            results = self.clasificacion_cnn(X_train, X_test, y_train, y_test)
            accs.append(results['test_accuracy'])
            losses.append(results['test_loss'])
            # accs.append(accuracy_score(y_test,y_pred))
            # recs.append(recall_score(y_test,y_pred,average='macro'))
            # precs.append(precision_score(y_test,y_pred,average='macro'))
            print(f"{k+1}/{n_iter} done...")
        return losses,accs
    
    def clasificacion_cnn(self,X_train, X_test, y_train, y_test): 
        val_size = int(0.85*X_train.shape[0])
        X_val = X_train[val_size:]
        y_val = y_train[val_size:]
        X_train = X_train[:val_size]
        y_train = y_train[:val_size]
        # print(X_train.shape,X_val.shape,X_test.shape)
        y_train = to_categorical(y_train,5)
        y_test = to_categorical(y_test,5)
        y_val = to_categorical(y_val,5)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
        X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], 1)
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_val = X_val.astype('float32')   
        model = self.__make_model(X_train.shape[1],X_train.shape[2])
        es = EarlyStopping(monitor='val_loss',patience=4)
        history = model.fit(X_train, y_train, batch_size=100, epochs=100, validation_data=(X_test, y_test),
                            callbacks=[es],verbose=0)
        score = model.evaluate(X_test, y_test)
        return {'test_loss': score[0], 'test_accuracy': score[1]}

    def __make_model(self,size_x,size_y):
        model = Sequential()
        model.add(Conv2D(25, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu', input_shape=(size_x,size_y,1)))
        model.add(MaxPool2D(pool_size=(1,1)))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dense(50, activation='relu'))
        if self.n_clases==2:
            model.add(Dense(1,activation='sigmoid'))
            model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam') # LOSS APROPIADA
        elif self.n_clases>2:
            model.add(Dense(self.n_clases, activation='softmax'))
            model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
        return model




class MyGridSearch:

    def __init__(self,param_dict:dict,hyper_params_dict):
        self.default_params_dict = {'top_n':100,
                                'alpha':0.9,
                                'beta1':1,
                                'beta2':1,
                                'n_cols':3
                                }
        all_params = list(self.default_params_dict.keys())
        if set(param_dict.keys()).issubset(set(all_params)):
            self.param_dict = param_dict
            self.best_params_dict = {k:np.nan for k in self.param_dict.keys()}
            self.best_accuracy = 0
            self.best_loss = 100
            self.hyper_params_dict = hyper_params_dict
            num_combinations = np.prod([len(x) for x in param_dict.values()])
            print(f"Number of combinations to try {num_combinations}")
        else:
            raise ValueError(f"Parametros no validos: {list(set(param_dict.keys()).difference(set(all_params)))}")

    def __get_kw(self,top_n):
        df = self.hyper_params_dict['df']
        df['Normalized Label'] = (1/2)*df['Sentiment'].values-1
        ake = KW(df=df,text_col_name="text",label_col_name="Normalized Label")
        return ake.get_kw(topn=top_n)

    def __do(self,combination:dict):
        '''
        Esta función realiza una "corrida" con una combinación de parámetros
        '''
        emb_model = self.hyper_params_dict['emb_model']
        kw_dict = self.__get_kw(top_n=combination['top_n'])
        df = self.hyper_params_dict['df']
        y = df['Sentiment'].values
        puntaje = Scoring(w2vmodel=emb_model,W0=kw_dict)
        puntaje.build_neighbors(alpha=combination['alpha'])
        scores_df = puntaje.transform(df=df,text_col="text",label_col="Normalized Label",
                                        beta1=combination['beta1'],
                                        beta2=combination['beta2'])
        # kw_dict = dict(zip(scores_df['word'].values,scores_df['score'].values))
        _ = puntaje.get_words_representations(mode='mean')
        X_msjs_rep = puntaje.get_texts_representations_MAT(cols_num=combination['n_cols'])
        # Hacer CV:
        X_train, X_test, y_train, y_test = train_test_split(X_msjs_rep, y, random_state=331,train_size=0.75) 
        results = self.clasificacion_cnn(X_train, X_test, y_train, y_test)
        if results['test_accuracy'] > self.best_accuracy:
            self.best_accuracy = results['test_accuracy']
            self.best_loss = results['test_loss']
            self.best_params_dict = combination
            print(f"Best combination so far: {combination}\nAccuracy:{self.best_accuracy}")

    def fit(self):
        '''
        Este método realiza la iteración sobre todos las combinaciones
        '''
        params = list(self.param_dict.keys())
        missing_params = [x for x in self.default_params_dict.keys() if x not in params]
        n_params = len(params)
        for values in product(*self.param_dict.values()):
            combination = {params[j]:values[j] for j in range(n_params)}
            # Llenar el resto de parametros con los defaults
            for x in missing_params:
                combination[x] = self.default_params_dict[x]
            # Realizar la corrida con estos parámetros
            print(f"Trying combination {combination}")
            self.__do(combination)
        # self.best_params = combination
        return self.best_params_dict

    def __make_model(self,size_x,size_y):
        n_clases =  self.hyper_params_dict['n_clases']
        model = Sequential()
        model.add(Conv2D(25, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu', input_shape=(size_x,size_y,1)))
        model.add(MaxPool2D(pool_size=(1,1)))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dense(50, activation='relu'))
        if n_clases==2:
            model.add(Dense(1,activation='sigmoid'))
            model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam') # LOSS APROPIADA
        elif n_clases>2:
            model.add(Dense(n_clases, activation='softmax'))
            model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
        return model

    def clasificacion_cnn(self,X_train, X_test, y_train, y_test): 
        val_size = int(0.85*X_train.shape[0])
        X_val = X_train[val_size:]
        y_val = y_train[val_size:]
        X_train = X_train[:val_size]
        y_train = y_train[:val_size]
        y_train = to_categorical(y_train,5)
        y_test = to_categorical(y_test,5)
        y_val = to_categorical(y_val,5)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
        X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], 1)
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_val = X_val.astype('float32')   
        model = self.__make_model(X_train.shape[1],X_train.shape[2])
        es = EarlyStopping(monitor='val_loss',patience=4)
        history = model.fit(X_train, y_train, batch_size=100, epochs=100, validation_data=(X_test, y_test),
                            callbacks=[es],verbose=0)
        score = model.evaluate(X_test, y_test)
        return {'test_loss': score[0], 'test_accuracy': score[1]}


class SentimentAnalysis:
    '''
    Esta clase encapsula todo el framework de la tarea de representación y clasificación para analisis de sentimientos, usando iteración o no. 
    Puede realizar el gridsearch para afinar parámetros.
    '''

    def __init__(self,hyper_params_dict,df,text_col_name='text',label_col_name='sentiment'):
        if text_col_name in df.columns.to_list() and text_col_name in df.columns.to_list():
            self.df = df
            self.text_col_name = text_col_name
            self.label_col_name = label_col_name
            self.hyper_params_dict = hyper_params_dict
            self.n_labels = np.unique(self.df[self.label_col_name].values).shape[0]
            print(f"Dataframe contains entries with {self.n_labels} labels:\n{np.unique(self.df[self.label_col_name].values)}")
        else:
            raise KeyError('No hay tales columnas en el dataframe')

    def __get_kw(self,top_n):
        self.df['Normalized Label'] = (2/(self.n_labels-1))*self.df[self.label_col_name].values-1
        ake = KW(df=self.df,
                    text_col_name=self.text_col_name,
                    label_col_name="Normalized Label")
        self.kw_dict = ake.get_kw(topn=top_n)
        return self.kw_dict

        # self.n_cols = n_cols
        # losses = []
        # accs = []
        # for k in range(n_iter):
        #     puntaje = Scoring(w2vmodel=self.emb_model,W0=self.kw_dict)
        #     puntaje.build_neighbors(alpha=alpha)
        #     scores_df = puntaje.transform(df=self.df,text_col="text",label_col="Normalized Label",beta1=beta1,beta2=beta2)
        #     kw_dict = dict(zip(scores_df['word'].values,scores_df['score'].values))
        #     _ = puntaje.get_words_representations(mode='mean')
        #     X_msjs_rep = puntaje.get_texts_representations_MAT(cols_num=n_cols)
        #     X_train, X_test, y_train, y_test = train_test_split(X_msjs_rep, self.y, random_state=331,train_size=0.75) 
        #     results = self.clasificacion_cnn(X_train, X_test, y_train, y_test)
        #     accs.append(results['test_accuracy'])
        #     losses.append(results['test_loss'])
        #     # accs.append(accuracy_score(y_test,y_pred))
        #     # recs.append(recall_score(y_test,y_pred,average='macro'))
        #     # precs.append(precision_score(y_test,y_pred,average='macro'))
        #     print(f"{k+1}/{n_iter} done...")
        # return losses,accs


    def __do(self,combination:dict,gs=True):
        '''
        Esta función realiza una "corrida" con una combinación de parámetros, 
        <gs>:   Si se usa como parte del método grid_search
        '''
        emb_model = self.hyper_params_dict['emb_model']
        y = self.df[self.label_col_name].values
        n_iter = combination['n_iter']
        if not gs:
            start = time.time()
            print("Calculando palabras prototípicas... ",end='')
        kw_dict = self.__get_kw(top_n=combination['top_n'])
        if not gs:
            print(f"done in {round(time.time()-start,5)}")
        for k in range(n_iter):
            if not gs:
                start = time.time()
                print("Construyendo vecinos... ", end='')
            puntaje = Scoring(w2vmodel=emb_model,W0=kw_dict)
            puntaje.build_neighbors(alpha=combination['alpha'])
            if not gs:
                print(f"done in {round(time.time()-start,5)}")
                start = time.time()
                print("Calculando puntajes... ", end='')
            scores_df = puntaje.transform(df=self.df,text_col=self.text_col_name,label_col="Normalized Label",
                                            beta1=combination['beta1'],
                                            beta2=combination['beta2'])
            if not gs:
                print(f"done in {round(time.time()-start,5)}")
                start = time.time()
                print("Calculando representaciones... ", end='')
            kw_dict = dict(zip(scores_df['word'].values,scores_df['score'].values))
            _ = puntaje.get_words_representations(mode='mean')
            X_msjs_rep = puntaje.get_texts_representations_MAT(cols_num=combination['n_cols'])
            # Hacer CV:
            if not gs:
                print(f"done in {round(time.time()-start,5)}")
                start = time.time()
                print("Entrenando clasificador... ", end='')
            X_train, X_test, y_train, y_test = train_test_split(X_msjs_rep, y, random_state=331,train_size=0.75) 
            results = self.__clasificacion_cnn(X_train, X_test, y_train, y_test)
            if not gs:
                print(f"done in {round(time.time()-start,5)}")
                print(f"Iteración {k+1}/{n_iter} completada.")
        if gs:
            if results['test_accuracy'] > self.best_accuracy:
                self.best_accuracy = results['test_accuracy']
                self.best_loss = results['test_loss']
                self.best_params_dict = combination
                print(f"Best combination so far: {combination}\nAccuracy:{self.best_accuracy}") 
        else: 
            return results

    def run(self,combination_dict,model=None):
        '''
        Correr todo el módelo de representación y clasificación para obtener las métricas de rendimiento, se usa
        una combinación de parámetros.
        '''
        if model is not None:
            self.model = model
        results = self.__do(combination_dict,gs=False)
        return results

    def __make_model(self,size_x,size_y):
        model = Sequential()
        model.add(Conv2D(25, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu', input_shape=(size_x,size_y,1)))
        model.add(MaxPool2D(pool_size=(1,1)))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dense(50, activation='relu'))
        if self.n_labels==2:
            model.add(Dense(1,activation='sigmoid'))
            model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam') # LOSS APROPIADA
        elif self.n_labels>2:
            model.add(Dense(self.n_labels, activation='softmax'))
            model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
        return model

    def __clasificacion_cnn(self,X_train, X_test, y_train, y_test): 
        val_size = int(0.85*X_train.shape[0])
        X_val = X_train[val_size:]
        y_val = y_train[val_size:]
        X_train = X_train[:val_size]
        y_train = y_train[:val_size]
        y_train = to_categorical(y_train,self.n_labels)
        y_test = to_categorical(y_test,self.n_labels)
        y_val = to_categorical(y_val,self.n_labels)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
        X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], 1)
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_val = X_val.astype('float32')   
        model = self.__make_model(X_train.shape[1],X_train.shape[2])
        es = EarlyStopping(monitor='val_loss',patience=4)
        history = model.fit(X_train, y_train, batch_size=100, epochs=100, validation_data=(X_test, y_test),
                            callbacks=[es],verbose=0)
        score = model.evaluate(X_test, y_test)
        y_pred = np.argmax(model(X_test),axis=1)  # para obtener las etiquetas predichas
        return {'test_loss': score[0], 'test_accuracy': score[1], 'predictions':y_pred}

    def grid_search(self,param_dict,default_params_dict):
        '''
        Este método realiza la iteración sobre todos las combinaciones
        '''
        self.param_dict = param_dict
        self.default_params_dict = default_params_dict
        params = list(self.param_dict.keys())
        missing_params = [x for x in self.default_params_dict.keys() if x not in params]
        n_params = len(params)
        num_combinations = np.prod([len(x) for x in param_dict.values()])
        print(f"Number of combinations to try {num_combinations}")
        self.best_accuracy = 0
        for values in product(*self.param_dict.values()):
            combination = {params[j]:values[j] for j in range(n_params)}
            # Llenar el resto de parametros con los defaults
            for x in missing_params:
                combination[x] = self.default_params_dict[x]
            # Realizar la corrida con estos parámetros
            print(f"Trying combination {combination}")
            self.__do(combination,gs=True)
        # self.best_params = combination
        return self.best_params_dict