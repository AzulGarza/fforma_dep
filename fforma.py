from tsfeatures import tsfeatures
from benchmarks import * 
import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import STL 
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from scipy.special import softmax
import copy

class FForma:
    def __init__(self):
        pass
    
    # Eval functions
    def smape(self, ts, ts_hat):
        num = np.abs(ts-ts_hat)
        den = np.abs(ts) + np.abs(ts_hat)
        return 2*np.mean(num/den)
    
    def mase(self, ts_train, ts_test, ts_hat):
        den = np.abs(np.diff(ts_train)).sum()/(len(ts_train) -1)
        return np.abs(ts_test - ts_hat).mean()/den

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
        self.models = basic_models
        self.ts_list = ts_list
        self.frcy = frcy
        self.fitted_models = [
            np.array([self.train_basic(model, ts, frcy) for model in basic_models]) for ts in ts_list
        ] 
        
        return self
        
    def predict_basic_models(self, h):
        self.h = h
        y_hat = [
            np.array([model.predict(h) for model in idts]) for idts in self.fitted_models
        ]
        
        return y_hat
    
    def _particular_error(self, tuple_test_pred, fun):
        ts_test_list, ts_pred_list = tuple_test_pred
        list_error = [fun(ts_test_list, pred) for pred in ts_pred_list]
        
        return np.array(list_error)
    
    def _particular_error_smape(self, tuple_test_pred):
        return self._particular_error(tuple_test_pred, self.smape)
    
    
    def _particular_error_mase(self, tuple_test_pred):
        return self._particular_error(tuple_test_pred, self.mase)
    
    def calculate_owa(self, ts_test_list, ts_hat_list, h, ts_train_list, frcy, parallel=True, threads=None):
        
        # init parallel
        if parallel and threads is None:
            threads = mp.cpu_count()
        
        if parallel:
            with mp.Pool(threads) as pool:
                
                ts_test_hat_list = zip(ts_test_list, ts_hat_list)

                smape_errors = pool.map(self._particular_error_smape, ts_test_hat_list)
                mase_errors = pool.map(self._particular_error_mase, ts_test_hat_list)
                
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
        
    
    def prepare_to_train(self, ts_test_list, ts_hat_list, parallel=True):
        ts_features = tsfeatures(self.ts_list, self.frcy, parallel=parallel)
        
        # Contribution to the owa error
        (_, contribution_to_owa) = self.calculate_owa(ts_test_list, ts_hat_list, self.h, self.ts_list, self.frcy, parallel=False)
        
        return (ts_features, contribution_to_owa.argmin(axis=1), contribution_to_owa)
    
     # Objective function for xgb
    def error_softmax_obj(self, predt: np.ndarray, dtrain: xgb.DMatrix) -> (np.ndarray, np.ndarray):
        '''
        Compute...
        '''
        y = dtrain.get_label()
        n_train = len(y)
        #print(predt.shape)
        preds_transformed = np.array([softmax(row) for row in predt])
        weighted_avg_loss_func = (preds_transformed*self.contribution_to_owa).sum(axis=1).reshape((n_train, 1))   
        grad = preds_transformed*(self.contribution_to_owa - weighted_avg_loss_func)
        hess = self.contribution_to_owa*preds_transformed*(1.0-preds_transformed) - grad*preds_transformed
        #print(grad)
        return grad.reshape(-1, 1), hess.reshape(-1, 1)

        
    def train(self, models, ts_list, frcy, val_periods=7, parallel=True):
        
        # Creating train and test sets
        ts_train_list = [ts[:(len(ts)-val_periods)] for ts in ts_list]
        ts_test_list = [ts[(len(ts)-val_periods):] for ts in ts_list]
        
        # Training and predict
        training = self.train_basic_models(models, ts_train_list, frcy)
        preds = training.predict_basic_models(val_periods)
        #print(preds)
        
        # Preparing data for xgb training
        self.X_train, self.y_train, self.contribution_to_owa = training.prepare_to_train(ts_test_list, preds)
        
        # For test purposes
        self.X_train = self.X_train
        
        # Training xgboost
        xgb_mat = xgb.DMatrix(data = self.X_train, label=self.y_train)
        
        param = {
            'max_depth': 3,  # the maximum depth of each tree
            'eta': 0.3,  # the training step for each iteration
            'silent': 1,  # logging mode - quiet
            'objective': 'multi:softprob',  # error evaluation for multiclass training
            'num_class': len(self.models),
            'nthread': 10
        }
        
        self.xgb = xgb.train(params=param, dtrain=xgb_mat, obj=self.error_softmax_obj, num_boost_round=10)
        
        # Training models with all data
        self.fitted_models = self.train_basic_models(models, ts_list, frcy).fitted_models
        
        # Optimal weights
        self.ts_feat = tsfeatures(ts_list, frcy, parallel=parallel)
        self.opt_weights = self.xgb.predict(xgb.DMatrix(self.ts_feat))
        
        return self
    
    def predict(self, h, ts_predict=None, frcy=None, parallel=True):
        """
        For each series in ts_list returns predictions
        ts_predict: list of series to predict
        """
        # Getting predictions for ts_predict
        if not (ts_predict is None):
            preds = self.train_basic_models(self.models, ts_predict, frcy).predict_basic_models(h)
            ts_feat = tsfeatures(ts_predict, frcy, parallel=parallel)
            opt_weights = self.xgb.predict(xgb.DMatrix(ts_feat))
            final_preds = final_preds = np.array([np.matmul(pred.T, opt_weight) for pred, opt_weight in zip(preds, opt_weights)])
        else:
            preds = np.array(self.predict_basic_models(h))
            #print(preds.shape)
            #print(self.opt_weights.shape)
            final_preds = np.array([np.matmul(pred.T, opt_weight) for pred, opt_weight in zip(preds, self.opt_weights)])
            
        return final_preds
        
        
        