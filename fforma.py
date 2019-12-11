from tsfeatures import tsfeatures
from benchmarks import *
import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import STL
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from scipy.special import softmax
import copy
import multiprocessing as mp
from sklearn.model_selection import train_test_split
import pickle
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
import tqdm


class FForma:
    def __init__(self):
        pass

    # Eval functions
    def smape(self, ts, ts_hat):
        # Needed condition
        assert ts.shape == ts_hat.shape, "ts must have the same size of ts_hat"

        num = np.abs(ts-ts_hat)
        den = np.abs(ts) + np.abs(ts_hat)
        return 2*np.mean(num/den)

    def mase(self, ts_train, ts_test, ts_hat, frcy):
        # Needed condition
        assert ts_test.shape == ts_hat.shape, "ts must have the same size of ts_hat"

        rolled_train = np.roll(ts_train, frcy)
        diff_train = np.abs(ts_train - rolled_train)
        den = diff_train[frcy:].mean()

        return np.abs(ts_test - ts_hat).mean()/den

    # Train functions
    def train_basic(self, model, ts, frcy):
        this_model = copy.deepcopy(model)
        if 'frcy' in model.fit.__code__.co_varnames:
            fitted_model = this_model.fit(ts, frcy)
        else:
            fitted_model = this_model.fit(ts)

        return fitted_model

    def train_basic_models(self, basic_models, ts_list, frcy):
        """
        basic_models: List of models
        """
        self.fitted_models = [
            np.array([self.train_basic(model, ts, frcy) for model in basic_models]) for ts in tqdm.tqdm(ts_list)
        ]

        return self

    def predict_basic_models(self, h):
        y_hat = [
            np.array([model.predict(h) for model in idts]) for idts in tqdm.tqdm(self.fitted_models)
        ]

        return y_hat
    # Auxiliars for parallel calculate owa
    def _particular_error(self, tuple_test_pred, fun):
        ts_test_list, ts_pred_list = tuple_test_pred
        list_error = [fun(ts_test_list, pred) for pred in ts_pred_list]

        return np.array(list_error)

    def _particular_error_smape(self, tuple_test_pred):
        return self._particular_error(tuple_test_pred, self.smape)

    def _simple_mase(self, ts_test, ts_hat):
        return np.abs(ts_test - ts_hat).mean()

    def _particular_error_simple_mase(self, tuple_test_pred):
        return self._particular_error(tuple_test_pred, self._simple_mase)

    def calculate_owa(self, ts_test_list, ts_hat_list, h, ts_train_list, frcy, parallel=True, threads=None, mult_preds=False):

        #To calculate owa correctly when ts_hat_list contains a single prediction
        if not mult_preds:
            ts_hat_list = np.array([ts_hat_list])

        # init parallel
        if parallel and threads is None:
            threads = mp.cpu_count()

        if parallel:
            with mp.Pool(threads) as pool:

                ts_test_hat_list = zip(ts_test_list, ts_hat_list)

                smape_errors = np.array(pool.map(self._particular_error_smape, ts_test_hat_list))
                mase_errors = np.array(pool.map(self._particular_error_simple_mase, ts_test_hat_list))
                den_mase = np.array([np.abs(np.diff(ts)).sum()/(len(ts) -1) for ts in ts_train_list])
                mase_errors = mase_errors/den_mase


                 ##### NAIVE2
                # Training naive2
                ts_hat_naive2 = [Naive2().fit(ts, frcy).predict(h) for ts in ts_train_list]

                # Smape of naive2
                mean_smape_naive2 = np.array([
                     self.smape(ts_test, ts_pred) for ts_test, ts_pred in zip(ts_test_list, ts_hat_naive2)
                ]).mean()

                print(mean_smape_naive2)

                # MASE of naive2
                mean_mase_naive2 = np.array([
                     self.mase(ts_train, ts_test, ts_pred) for ts_train, ts_test, ts_pred in zip(ts_train_list, ts_test_list, ts_hat_naive2)
                ]).mean()

                print(mean_mase_naive2)

        else:

            smape_errors = np.array([
                np.array(
                    [self.smape(ts_test, pred) for pred in ts_pred]
                ) for ts_test, ts_pred in zip(ts_test_list, ts_hat_list)
            ])


            # Mase
            mase_errors = np.array([
                np.array(
                    [self.mase(ts_train, ts_test, pred) for pred in ts_pred]
                ) for ts_train, ts_test, ts_pred in zip(ts_train_list, ts_test_list, ts_hat_list)
            ])

            ##### NAIVE2
            # Training naive2
            ts_hat_naive2 = [Naive2().fit(ts, frcy).predict(h) for ts in ts_train_list]

            # Smape of naive2
            mean_smape_naive2 = np.array([
                 self.smape(ts_test, ts_pred) for ts_test, ts_pred in zip(ts_test_list, ts_hat_naive2)
            ]).mean()

            # MASE of naive2
            mean_mase_naive2 = np.array([
                 self.mase(ts_train, ts_test, ts_pred) for ts_train, ts_test, ts_pred in zip(ts_train_list, ts_test_list, ts_hat_naive2)
            ]).mean()

        # Contribution to the owa error
        contribution_to_owa = (smape_errors/mean_smape_naive2) + (mase_errors/mean_mase_naive2)
        contribution_to_owa = contribution_to_owa/2

        return (contribution_to_owa.mean(), contribution_to_owa)


    def prepare_to_train(self, ts_list, ts_test_list, ts_hat_list, h, frcy, parallel=True):
        ts_features = tsfeatures(ts_list, frcy, parallel=parallel)

        # Contribution to the owa error
        (_, contribution_to_owa) = self.calculate_owa(ts_test_list, ts_hat_list, h, ts_list, frcy, parallel=False, mult_preds=True)

        return (ts_features, contribution_to_owa.argmin(axis=1), contribution_to_owa)

     # Objective function for xgb
    def error_softmax_obj(self, predt: np.ndarray, dtrain: xgb.DMatrix) -> (np.ndarray, np.ndarray):
        '''
        Compute...
        '''
        y = dtrain.get_label().astype(int)
        #print(y)
        n_train = len(y)
        #print(predt.shape)
        #print(predt)
        preds_transformed = predt#np.array([softmax(row) for row in predt])
        weighted_avg_loss_func = (preds_transformed*self.contribution_to_owa[y, :]).sum(axis=1).reshape((n_train, 1))
        grad = preds_transformed*(self.contribution_to_owa[y, :] - weighted_avg_loss_func)
        hess = self.contribution_to_owa[y,:]*preds_transformed*(1.0-preds_transformed) - grad*preds_transformed
        #print(grad)
        return grad.reshape(-1, 1), hess.reshape(-1, 1)

    def fforma_loss(self, predt: np.ndarray, dtrain: xgb.DMatrix) -> (str, float):
        '''
        Compute...
        '''
        y = dtrain.get_label().astype(int)
        n_train = len(y)
        #print(predt.shape)
        #print(predt)
        preds_transformed = predt#np.array([softmax(row) for row in predt])
        weighted_avg_loss_func = (preds_transformed*self.contribution_to_owa[y, :]).sum(axis=1).reshape((n_train, 1))
        fforma_loss = weighted_avg_loss_func.sum()
        #print(grad)
        return 'FFORMA-loss', fforma_loss

    def forec_methods(self):
        return [Naive(), SeasonalNaive(), RandomWalkDrift(), ETS()]

    def _preprocess_train(self, ts_train_list, val_periods, h, frcy, models, save_file=None):
        """
        Make all necessary preprocessing to train an fforma model.
        returns a dict
        ts_train_list:  list where each element is a numpy array fot each time series
        """
        dict_return = {}
        # Init definitions
        dict_return['frcy'] = frcy
        dict_return['test_periods'] = h
        #Number of models (number of classes for xgboost)
        dict_return['n_models'] = len(models)
        #Split in train/validation sets
        train = [ts[:-val_periods] for ts in ts_train_list]
        validation = [ts[-val_periods:] for ts in ts_train_list]
        dict_return['train'] = train
        dict_return['validation'] = validation

        print('Making predictions for validation set')
        preds_validation = self.train_basic_models(models, train, frcy).predict_basic_models(val_periods)
        dict_return['preds_validation'] = preds_validation

        print('Making predictions for model')
        preds_h = self.train_basic_models(models, ts_train_list, frcy).predict_basic_models(h)
        dict_return['preds_h'] = preds_h

        print('Calculating features')
        train_feats = tsfeatures(train, frcy, parallel=True)
        dict_return['train_feats'] = train_feats

        print('Calculating complete features')
        train_feats_complete = tsfeatures(ts_train_list, frcy, parallel=True)
        dict_return['train_feats_complete'] = train_feats_complete

        print('Calculating contribution_to_owa with validation set')
        (_, contribution_to_owa) = self.calculate_owa(
            validation,
            preds_validation,
            val_periods,
            train,
            frcy,
            parallel=False,
            mult_preds=True
        )

        dict_return['contribution_to_owa'] = contribution_to_owa
        dict_return['best_model'] = contribution_to_owa.argmin(axis=1)

        if save_file is None:
            return dict_return
        else:
            dict_return['save_file'] = save_file
            with open(save_file, 'wb') as file:
                pickle.dump(dict_return, file, protocol=pickle.HIGHEST_PROTOCOL)

            print(f"Saved preprocessing to {save_file}")

    # Functions for training xgboost
    def _train_xgboost(self, params):

        gbm_model = xgb.train(
            params=params,
            dtrain=self.dtrain,
            obj=self.error_softmax_obj,
            num_boost_round=999,
            feval=self.fforma_loss,
            evals=[(self.dtrain, 'eval'), (self.dvalid, 'train')],
            early_stopping_rounds=99,
            verbose_eval = False
        )

        return gbm_model

    def _score(self, params):

        #training model
        gbm_model = self._train_xgboost(params)

        predictions = gbm_model.predict(
            self.dvalid,
            ntree_limit=gbm_model.best_iteration + 1#,
            #output_margin = True
        )

        #print(predictions)

        loss = self.fforma_loss(predictions, self.dvalid)
        # TODO: Add the importance for the selected features
        #print("\tLoss {0}\n\n".format(loss))

        return_dict = {'loss': loss[1], 'status': STATUS_OK}

        return return_dict

    def _optimize_xgb(self, threads, random_state, max_evals):
        """
        This is the optimization function that given a space (space here) of
        hyperparameters and a scoring function (score here), finds the best hyperparameters.
        """
        # To learn more about XGBoost parameters, head to this page:
        # https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
        space = {
            'n_estimators': hp.quniform('n_estimators', 100, 1000, 1),
            'eta': hp.quniform('eta', 0.025, 0.5, 0.025),
            # A problem with max_depth casted to float instead of int with
            # the hp.quniform method.
            'max_depth':  hp.choice('max_depth', np.arange(1, 14, dtype=int)),
            'min_child_weight': hp.quniform('min_child_weight', 1, 6, 1),
            'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
            'gamma': hp.quniform('gamma', 0.5, 1, 0.05),
            'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05)
        }
        space = {**space, **self.init_params}
        # Use the fmin function from Hyperopt to find the best hyperparameters
        best = fmin(self._score, space, algo=tpe.suggest,
                    # trials=trials,
                    max_evals=max_evals)
        return best

    def _wrapper_best_xgb(self, threads, random_state, max_evals):

        # Optimizing xgbost
        best_hyperparameters = self._optimize_xgb(threads, random_state, max_evals)

        best_hyperparameters = {**best_hyperparameters, **self.init_params}

        # training optimal xgboost model
        gbm_best_model = self._train_xgboost(best_hyperparameters)

        return gbm_best_model

    def _train_from_file(self, file, random_state, threads=None, max_evals=100, save_file=False):
        """
        Train xgboost with randomized
        """
        with open(file, 'rb') as read_file:
            prep_dict = pickle.load(read_file)

        # nthreads params
        if threads is None:
            threads = mp.cpu_count()

        self.n_models = prep_dict['n_models']
        self.contribution_to_owa = prep_dict['contribution_to_owa']

        # Train-validation sets for XGBoost
        X_train_xgb, X_val, y_train_xgb, \
            y_val, indices_train, \
            indices_val = train_test_split(
                prep_dict['train_feats'],
                prep_dict['best_model'],
                np.arange(prep_dict['train_feats'].shape[0])
            )

        self.dtrain = xgb.DMatrix(data=X_train_xgb, label=indices_train)
        self.dvalid = xgb.DMatrix(data=X_val, label=indices_val)

        self.init_params = {
            'objective': 'multi:softprob',
            # Increase this number if you have more cores. Otherwise, remove it and it will default
            # to the maxium number.
            'num_class': self.n_models,
            'nthread': threads,
            #'booster': 'gbtree',
            #'tree_method': 'exact',
            'silent': 1,
            'seed': random_state,
            'disable_default_eval_metric': 1
        }

        self.xgb = self._wrapper_best_xgb(threads, random_state, max_evals)

        if save_file:
            prep_dict['xgb'] = self.xgb
            #prep_dict['dtrain'] = self.dtrain
            with open(prep_dict['save_file'], 'wb') as file:
                pickle.dump(prep_dict, file, protocol=pickle.HIGHEST_PROTOCOL)

            print(f"Saved model to {prep_dict['save_file']}")
        else:
            return self


    def _predict_from_file(self, file):
        with open(file, 'rb') as read_file:
            train_dict = pickle.load(read_file)

        xgb_ = train_dict['xgb']
        ts_feats = xgb.DMatrix(train_dict['train_feats'])

        preds = train_dict['preds_h']
        opt_weights = xgb_.predict(ts_feats)

        preds = np.array(
            [np.matmul(pred.T, opt_weight) for pred, opt_weight in zip(preds, opt_weights)]
        )

        return preds
    # Tranfer learning
    def tranfer_learning(self, file1, file2):
        """
        file1: must be trained
        file2: must be preprocessed data
        """
        with open(file1, 'rb') as read_file:
            to_transfer = pickle.load(read_file)

        with open(file2, 'rb') as read_file:
            to_receive = pickle.load(read_file)

        xgb_ = to_transfer['xgb']
        # TO receive
        ts_feats_recieve = xgb.DMatrix(to_receive['train_feats_complete'])
        preds_receive = to_receive['preds_h']
        opt_weights_receive = xgb_.predict(ts_feats_recieve)

        preds = np.array(
            [np.matmul(pred.T, opt_weight) for pred, opt_weight \
            in zip(preds_receive, opt_weights_receive)]
        )

        return preds

    def train(self, models, ts_list, frcy, val_periods=7, parallel=True, threads=None):

        # Defining self values
        self.models = basic_models
        self.ts_list = ts_list
        self.frcy = frcy

        # Creating train and test sets
        ts_train_list = [ts[:(len(ts)-val_periods)] for ts in ts_list]
        ts_test_list = [ts[(len(ts)-val_periods):] for ts in ts_list]

        # Training and predict
        training = self.train_basic_models(models, ts_train_list, frcy)
        preds = training.predict_basic_models(val_periods)
        #print(preds)

        # Preparing data for xgb training
        self.X_train, self.y_train, self.contribution_to_owa = training.prepare_to_train(ts_train_list, ts_test_list, preds, val_periods, frcy)
        #self.X_val =  tsfeatures(ts_test_list, frcy, parallel=parallel)
        indexes = np.arange(self.X_train.shape[0])
        X_train_xgb, X_val, y_train_xgb, y_val, indices_train, indices_val = train_test_split(self.X_train, self.y_train, indexes)


        # Training xgboost
        xgb_mat = xgb.DMatrix(data=X_train_xgb, label=indices_train)
        xgb_mat_val = xgb.DMatrix(data=X_val, label=indices_val)

        # nthreads params
        if threads is None:
            threads = mp.cpu_count()


        param = {
            'max_depth': 3,  # the maximum depth of each tree
            'eta': 0.3,  # the training step for each iteration
            'silent': 1,  # logging mode - quiet
            'objective': 'multi:softprob',  # error evaluation for multiclass training
            'num_class': len(self.models),
            'nthread': threads,
            'disable_default_eval_metric': 1
        }

        self.xgb = xgb.train(
            params=param,
            dtrain=xgb_mat,
            obj=self.error_softmax_obj,
            num_boost_round=100,
            feval=self.fforma_loss,
            evals=[(xgb_mat, 'dtrain'), (xgb_mat_val, 'dval')],
            early_stopping_rounds=10
        )

        # Training models with all data
        self.fitted_models = self.train_basic_models(models, ts_list, frcy).fitted_models

        # Optimal weights
        self.ts_feat = tsfeatures(ts_list, frcy, parallel=parallel)
        self.ts_feat = self.ts_feat[self.xgb.feature_names]

        self.opt_weights = self.xgb.predict(xgb.DMatrix(self.ts_feat))

        return self

    def predict(self, h, ts_predict=None, frcy=None, parallel=True):
        """
        For each series in ts_list returns predictions
        ts_predict: list of series to predict
        """
        self.h = h
        # Getting predictions for ts_predict
        if not (ts_predict is None):
            preds = self.train_basic_models(self.models, ts_predict, frcy).predict_basic_models(h)
            ts_feat = tsfeatures(ts_predict, frcy, parallel=parallel)
            ts_feat = ts_feat[self.xgb.feature_names]
            opt_weights = self.xgb.predict(xgb.DMatrix(ts_feat))
            final_preds = np.array([np.matmul(pred.T, opt_weight) for pred, opt_weight in zip(preds, opt_weights)])
        else:
            preds = np.array(self.predict_basic_models(h))
            #print(preds.shape)
            #print(self.opt_weights.shape)
            final_preds = np.array([np.matmul(pred.T, opt_weight) for pred, opt_weight in zip(preds, self.opt_weights)])

        return final_preds
