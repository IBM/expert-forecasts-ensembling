from scipy.optimize import minimize
from cvxopt import matrix, solvers
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

import numpy as np
import pandas as pd

def simple_average(task, *argv):

    characteristics = argv[0].copy()

    if task == "train":
       mdl = {'name' : 'simple_average', 'characteristics' : characteristics}
       return mdl

    elif task == "score":
       X = argv[1]
       return np.nanmean(X, axis=1)
       

def trimmed_average(task, *argv):

    characteristics = argv[0].copy()
    X = argv[1]

    if len(characteristics['baseLearners']) <= 2: 
       raise Exception("For 'trimmed_average' method, at least three base models are required")

    if task == "train":
       mdl = {'name' : 'trimmed_average', 'characteristics' : characteristics}
       return mdl

    elif task == "score":
       return (np.sum(X, axis=1) - np.amax(X, axis=1) - np.amin(X, axis=1)) / (X.shape[1] - 2)


def windsorized_average(task, *argv):

    characteristics = argv[0].copy()    

    if task == "train":
       mdl = {'name' : 'windsorized_average', 'characteristics' : characteristics}
       return mdl


    elif task == "score":
       X = argv[1]
       return (np.sum(X, axis=1) - np.amax(X, axis=1) - np.amin(X, axis=1) + np.sort(X)[:, 1] + np.sort(X)[:, -2]) / (X.shape[1])


def ols_average(task, *argv):

    characteristics = argv[0].copy()
    X = argv[1]

    if 'bias' not in characteristics:
       characteristics['bias'] = False

    if characteristics['bias']:
       X = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)

    if task == "train":
       y = argv[2]
       weights = np.linalg.lstsq(X, y, rcond=None)[0]      
       characteristics['weights'] = weights.tolist()       
       mdl = {'name' : 'ols_average', 'characteristics' : characteristics}
       return mdl


    elif task == "score":
       weights = characteristics['weights']
       return np.dot(X, np.asarray(weights))


def lad_average(task, *argv):

    characteristics = argv[0].copy()
    X = argv[1]

    if 'bias' not in characteristics:
       characteristics['bias'] = False

    if characteristics['bias']:
       X = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)

    if task == "train":
       y = argv[2]
       weights = np.linalg.lstsq(X, y, rcond=None)[0]  
       weights = minimize(LADcostfun, weights, args=(X, y)).x   
       characteristics['weights'] = weights.tolist()      
       mdl = {'name' : 'lad_average', 'characteristics' : characteristics}
       return mdl


    elif task == "score":
       weights = characteristics['weights']
       return np.dot(X, np.asarray(weights))
       

def pw_average(task, *argv):

    characteristics = argv[0].copy()
    X = argv[1]

    if task == "train":
       y = argv[2]
       Q = 2 * matrix(np.dot(X.T, X))
       p = matrix(-2 * np.dot(X.T, y))
       G = matrix(-1 * np.identity(X.shape[1]))
       h = matrix(np.zeros(X.shape[1]))
       sol = solvers.qp(Q, p, G, h)
       weights = np.asarray(sol['x']).ravel()
       characteristics['weights'] = weights.tolist()      
       mdl = {'name' : 'pw_average', 'characteristics' : characteristics}
       return mdl


    elif task == "score":
       weights = characteristics['weights']
       return np.dot(X, np.asarray(weights))


def cls_average(task, *argv):

    characteristics = argv[0].copy()
    X = argv[1]

    if task == "train":
       y = argv[2]
       Q = 2 * matrix(np.dot(X.T, X))
       p = matrix(-2 * np.dot(X.T, y))
       G = matrix(np.concatenate((np.identity(X.shape[1]),-1 * np.identity(X.shape[1])), axis=0))
       h = matrix(np.concatenate((np.ones((X.shape[1], 1)), np.zeros((X.shape[1], 1))), axis=0))
       A = matrix(np.ones((1, X.shape[1])))
       b = matrix(1.0)
       sol = solvers.qp(Q, p, G, h, A, b)
       weights = np.asarray(sol['x']).ravel()
       characteristics['weights'] = weights.tolist()       
       mdl = {'name' : 'cls_average', 'characteristics' : characteristics}
       return mdl


    elif task == "score":
       weights = characteristics['weights']
       return np.dot(X, np.asarray(weights))


def mlp_regression(task, *argv):

    characteristics = argv[0].copy()
    X = argv[1].copy()

    if task == "train":
       y = argv[2].copy()
       Xy = np.concatenate((X,np.reshape(y, (-1,1))), axis=1)
       scaler = StandardScaler()
       scaler.fit(Xy)
       Xy_scale = (Xy - scaler.mean_)/scaler.scale_
       X_scale = Xy_scale[:,:-1]
       y_scale = Xy_scale[:,-1]

       mlp = MLPRegressor(alpha=characteristics['alpha'], activation=characteristics['activation'], hidden_layer_sizes=characteristics['hidden_layer_sizes'])
       mlp.fit(X_scale, y_scale)

       characteristics['scaler'] = scaler       
       characteristics['fitted_model'] = mlp       
       mdl = {'name' : 'mlp_regression', 'characteristics' : characteristics}
       return mdl


    elif task == "score":
       scaler = characteristics['scaler']
       mlp = characteristics['fitted_model']
       X_scale = (X-scaler.mean_[:-1])/scaler.scale_[:-1]
       y_predict_scale = mlp.predict(X_scale)
       return (y_predict_scale*scaler.scale_[-1]) + scaler.mean_[-1]


def irmse_average(task, *argv):
 
    characteristics = argv[0].copy()

    if task == "train":
       characteristics['is_recursive'] = True
       mdl = {'name' : irmse_average.__name__, 'characteristics' : characteristics}
       return mdl

    elif task == "score":      
       X = argv[1]
       X_train = characteristics['X_train']
       y_train = characteristics['y_train']
    
       RMSE_i = []
       for i in range(X_train.shape[1]):
           RMSE_i.append(RMSEfun(y_train, X_train[:,i]))

       IRMSE_i = [1/el for el in RMSE_i]
       IRMSE_sum = sum(IRMSE_i)
       weights = [el/IRMSE_sum for el in IRMSE_i]
       return np.dot(X, np.asarray(weights))


def sc_fixed_share_aggregation(task, *argv):

    if task == "train":
       X = argv[1].copy()
       y = argv[2].copy()
       Xy = np.concatenate((X,np.reshape(y, (-1,1))), axis=1)
       scaler = StandardScaler()
       scaler.fit(Xy)
       characteristics = argv[0].copy()
       characteristics['scaler'] = scaler
       characteristics['is_recursive'] = True
       characteristics['eta'] = characteristics['initialization']['eta']
       characteristics['alpha'] = characteristics['initialization']['alpha']
       characteristics['weights'] = characteristics['initialization']['weights']
       mdl = {'name' : sc_fixed_share_aggregation.__name__, 'characteristics' : characteristics}
       return mdl

    elif task == "score":
       characteristics = argv[0]
       scaler = characteristics['scaler']
       ###PREVIOUS PREDICTION EVALUATION
       weights_previous = np.asarray(characteristics['weights'])
       X_train = characteristics['X_train']
       y_train = characteristics['y_train']
       Xy_train = np.concatenate((X_train,np.reshape(y_train, (-1,1))), axis=1)
       Xy_train_scale = (Xy_train - scaler.mean_)/scaler.scale_
       X_train_scale = Xy_train_scale[:,:-1]
       y_train_scale = Xy_train_scale[:,-1]
       last_val_ind = pd.DataFrame(y_train_scale).apply(pd.Series.last_valid_index).values[0]
       y_predict_scale_previous = (1/np.sum(weights_previous))*np.dot(X_train_scale[last_val_ind,:].ravel(), weights_previous)
       
       ###LOSS UPDATE
       eta = characteristics['eta'] 
       active_set_previous = np.argwhere(~np.isnan(X_train[last_val_ind,:]))
       exp_term = np.exp(-eta*squareLossfun(y_train_scale[last_val_ind], X_train_scale[last_val_ind,:]))
       vi = np.empty(X_train.shape[1])
       vi[:] = np.nan
       vi[active_set_previous] = np.multiply(weights_previous[active_set_previous], exp_term[active_set_previous])

       ###SHARE UPDATE
       alpha = characteristics['alpha']
       X = argv[1]
       X_scale = (X-scaler.mean_[:-1])/scaler.scale_[:-1]
       active_set = np.argwhere(~np.isnan(X[0,:])) 
       active_previous_notnow_set = np.setdiff1d(active_set_previous, active_set) 
       active_previous_andnow_set = np.intersect1d(active_set_previous, active_set)
       cardinality_active_set = len(active_set)
       weights = np.zeros(X_train.shape[1])
       weights[active_set] = (1/cardinality_active_set)*np.sum(vi[active_previous_notnow_set]) + (alpha/cardinality_active_set)*np.sum(vi[active_previous_andnow_set])
       weights[active_previous_andnow_set] = weights[active_previous_andnow_set] + (1 - alpha)*vi[active_previous_andnow_set]
       #print('y_train_last: ' + str(y_train[last_val_ind]))
       #print('X_train_last: ' + str(X_train[last_val_ind,:]))

       ###NEW PREDICTION
       y_predict_scale = (1/np.sum(weights))*np.dot(X_scale[0,active_set].ravel(), weights[active_set].ravel())
       y_predict = (y_predict_scale*scaler.scale_[-1]) + scaler.mean_[-1]

       ###UPDATE WEIGHTS ARRAY IN THE MODEL DICTIONARY
       characteristics['weights'] = weights.tolist().copy()

       #print('active set previous: ' + str(active_set_previous))
       #print('active_set: ' + str(active_set))
       #print('weights_previous: ' + str(weights_previous))
       #print(characteristics['baseLearners'])
       #print('weights: ' + str(weights/np.sum(weights)))
       #print('cardinality_active_set: ' + str(cardinality_active_set))
       #print('active_previous_notnow_set: ' + str(active_previous_notnow_set))
       #print('exp_term: ' + str(exp_term))
       #print('vi: ' + str(vi))
       #print('loss ' + str(squareLossfun(y_train_scale[last_val_ind], X_train_scale[last_val_ind,:])))
       #print('A: ' + str((1/cardinality_active_set)*np.sum(vi[active_previous_notnow_set])))
       #print('B: ' + str((alpha/cardinality_active_set)*np.sum(vi[active_previous_andnow_set])))
       #print('C: ' + str((1 - alpha)*vi[active_previous_andnow_set]))
       #print('y_predict: ' + str(y_predict))

       for t in range(1, X.shape[0]):

           ###SHARE UPDATE
           weights_previous = weights.copy()
           active_set_previous = active_set.copy()
           active_set = np.argwhere(~np.isnan(X[t,:])) 
           active_previous_notnow_set = np.setdiff1d(active_set_previous, active_set) 
           active_previous_andnow_set = np.intersect1d(active_set_previous, active_set)
           cardinality_active_set = len(active_set)
           weights = np.zeros(X.shape[1])
           weights[active_set] = (1/cardinality_active_set)*np.sum(weights_previous[active_previous_notnow_set]) + (alpha/cardinality_active_set)*np.sum(weights_previous[active_previous_andnow_set])
           weights[active_previous_andnow_set] = weights[active_previous_andnow_set] + (1 - alpha)*weights_previous[active_previous_andnow_set]

           ###NEW PREDICTION
           y_predict_scale_new = (1/np.sum(weights))*np.dot(X_scale[t,active_set].ravel(), weights[active_set].ravel())
           y_predict_new = (y_predict_scale_new*scaler.scale_[-1]) + scaler.mean_[-1]
           y_predict = np.append(y_predict, y_predict_new)
        
           #print('active set previous: ' + str(active_set_previous))
           #print('active_set: ' + str(active_set))
           #print('weights_previous: ' + str(weights_previous))
           #print(characteristics['baseLearners'])
           #print('weights: ' + str(weights/np.sum(weights)))
           #print('cardinality_active_set: ' + str(cardinality_active_set))
           #print('active_previous_notnow_set: ' + str(active_previous_notnow_set))
           #print('A2: ' + str((1/cardinality_active_set)*np.sum(weights_previous[active_previous_notnow_set])))
           #print('B2: ' + str((alpha/cardinality_active_set)*np.sum(weights_previous[active_previous_andnow_set])))
           #print('C2: ' + str((1 - alpha)*weights_previous[active_previous_andnow_set]))
           #print('y_predict: ' + str(y_predict))

       return y_predict


def gam_sc_fixed_share_aggregation(task, *argv):

    if task == "train":
       characteristics = argv[0].copy()
       characteristics['is_recursive'] = True
       characteristics['eta'] = characteristics['initialization']['eta']
       characteristics['alpha'] = characteristics['initialization']['alpha']
       characteristics['weights'] = characteristics['initialization']['weights']
       mdl = {'name' : gam_sc_fixed_share_aggregation.__name__, 'characteristics' : characteristics}
       return mdl

    elif task == "score":
       characteristics = argv[0]
       ###PREVIOUS PREDICTION EVALUATION
       weights_previous = np.asarray(characteristics['weights'])
       scaler = characteristics['scaler']
       X_train = characteristics['X_train']
       y_train = characteristics['y_train']
       last_val_ind = pd.DataFrame(y_train).apply(pd.Series.last_valid_index).values[0]
       
       y_predict_scale_previous = len(weights_previous)*(1/np.sum(weights_previous))*np.dot(X_train[last_val_ind,:].ravel(), weights_previous)
       
       ###LOSS UPDATE
       eta = characteristics['eta'] 
       active_set_previous = np.argwhere(~np.isnan(X_train[last_val_ind,:]))
       exp_term = np.exp(-eta*squareLossfun(y_train[last_val_ind], X_train[last_val_ind,:]))
       vi = np.empty(X_train.shape[1])
       vi[:] = np.nan
       vi[active_set_previous] = np.multiply(weights_previous[active_set_previous], exp_term[active_set_previous])

       ###SHARE UPDATE
       alpha = characteristics['alpha']
       X = argv[1]
       active_set = np.argwhere(~np.isnan(X[0,:])) 
       active_previous_notnow_set = np.setdiff1d(active_set_previous, active_set) 
       active_previous_andnow_set = np.intersect1d(active_set_previous, active_set)
       cardinality_active_set = len(active_set)
       weights = np.zeros(X_train.shape[1])
       weights[active_set] = (1/cardinality_active_set)*np.sum(vi[active_previous_notnow_set]) + (alpha/cardinality_active_set)*np.sum(vi[active_previous_andnow_set])
       weights[active_previous_andnow_set] = weights[active_previous_andnow_set] + (1 - alpha)*vi[active_previous_andnow_set]
       #print('y_train_last: ' + str(y_train[last_val_ind]))
       #print('X_train_last: ' + str(X_train[last_val_ind,:]))

       ###NEW PREDICTION 
       y_predict_scale = len(weights)*(1/np.sum(weights))*np.dot(X[0,active_set].ravel(), weights[active_set].ravel())
       y_predict = (y_predict_scale*scaler.scale_[-1]) + scaler.mean_[-1]

       ###UPDATE WEIGHTS ARRAY IN THE MODEL DICTIONARY
       characteristics['weights'] = weights.tolist().copy()

       #print('active set previous: ' + str(active_set_previous))
       #print('active_set: ' + str(active_set))
       #print('weights_previous: ' + str(weights_previous))
       print(characteristics['baseLearners'])
       print('weights: ' + str(weights))
       #print('cardinality_active_set: ' + str(cardinality_active_set))
       #print('active_previous_notnow_set: ' + str(active_previous_notnow_set))
       print('exp_term: ' + str(exp_term))
       print('vi: ' + str(vi))
       print('loss ' + str(squareLossfun(y_train[last_val_ind], X_train[last_val_ind,:])))
       #print('A: ' + str((1/cardinality_active_set)*np.sum(vi[active_previous_notnow_set])))
       #print('B: ' + str((alpha/cardinality_active_set)*np.sum(vi[active_previous_andnow_set])))
       #print('C: ' + str((1 - alpha)*vi[active_previous_andnow_set]))
       #print('y_predict: ' + str(y_predict))

       for t in range(1, X.shape[0]):

           ###SHARE UPDATE
           weights_previous = weights.copy()
           active_set_previous = active_set.copy()
           active_set = np.argwhere(~np.isnan(X[t,:])) 
           active_previous_notnow_set = np.setdiff1d(active_set_previous, active_set) 
           active_previous_andnow_set = np.intersect1d(active_set_previous, active_set)
           cardinality_active_set = len(active_set)
           weights = np.zeros(X.shape[1])
           weights[active_set] = (1/cardinality_active_set)*np.sum(weights_previous[active_previous_notnow_set]) + (alpha/cardinality_active_set)*np.sum(weights_previous[active_previous_andnow_set])
           weights[active_previous_andnow_set] = weights[active_previous_andnow_set] + (1 - alpha)*weights_previous[active_previous_andnow_set]

           ###NEW PREDICTION
           y_predict_scale_new = len(weights)*(1/np.sum(weights))*np.dot(X[t,active_set].ravel(), weights[active_set].ravel())
           y_predict_new = (y_predict_scale_new*scaler.scale_[-1]) + scaler.mean_[-1]
           y_predict = np.append(y_predict, y_predict_new)
        
           #print('active set previous: ' + str(active_set_previous))
           #print('active_set: ' + str(active_set))
           #print('weights_previous: ' + str(weights_previous))
           print(characteristics['baseLearners'])
           print('weights: ' + str(weights))
           #print('cardinality_active_set: ' + str(cardinality_active_set))
           #print('active_previous_notnow_set: ' + str(active_previous_notnow_set))
           #print('A2: ' + str((1/cardinality_active_set)*np.sum(weights_previous[active_previous_notnow_set])))
           #print('B2: ' + str((alpha/cardinality_active_set)*np.sum(weights_previous[active_previous_andnow_set])))
           #print('C2: ' + str((1 - alpha)*weights_previous[active_previous_andnow_set]))
           #print('y_predict: ' + str(y_predict))

       return y_predict


#################### COST FUNCTIONS
def LADcostfun(params, X, y):
    return np.sum(np.abs(y - np.dot(X, params)))

def RMSEfun(target, forecast):
    return np.sqrt(np.nanmean(np.power((target - forecast),2)))

#################### LOSS FUNCTIONS
def squareLossfun(target, forecast):
    return np.power((target - forecast),2)

