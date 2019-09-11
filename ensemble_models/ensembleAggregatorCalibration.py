from scipy.optimize import minimize
from cvxopt import matrix, solvers
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import make_scorer

import numpy as np


def simple_average(task, *argv):

    characteristics = argv[0].copy()

    if task == "train":
       mdl = {'name' : 'simple_average', 'characteristics' : characteristics}
       return mdl

    elif task == "score":
       X = argv[1]
       return np.mean(X, axis=1)
       

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

       iterations = 200
       mlp = MLPRegressor(max_iter=iterations)
       ####################### CALIBRATION SECTION
       tcvg = TimeSeriesSplit(max_train_size=None, n_splits=5)
       param_grid = {'activation': ['relu', 'logistic', 'identity', 'tanh'],
                     'alpha': [0.001, 0.01, 0.1, 1, 10],
                     'hidden_layer_sizes': [(5,), (10,), (100,), (200,)]
                    }
       mlp_calibration = GridSearchCV(estimator=mlp,
                                      param_grid=param_grid,
                                      cv=tcvg,
                                      scoring=make_scorer(RMSEfun,
                                                          greater_is_better=False),
                                                          refit=True)

       ########################
       mlp_calibration.fit(X_scale, y_scale)
       print("Creating a neural network with " +
       str(len(mlp_calibration.best_estimator_.hidden_layer_sizes)) + " layers of size " + str(mlp_calibration.best_estimator_.hidden_layer_sizes) + " , " +
       str(iterations) + " iterations, " + "alpha = " +
       str(mlp_calibration.best_estimator_.alpha) + " and " +
       mlp_calibration.best_estimator_.activation + " activation function")
       mlp = mlp_calibration.best_estimator_
       #mlp.fit(X_scale, y_scale)

       characteristics['scaler'] = scaler       
       characteristics['fitted_model'] = mlp       
       characteristics['calibration_results'] = mlp_calibration
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
       
           
#################### COST FUNCTIONS
def LADcostfun(params, X, y):
    return np.sum(np.abs(y - np.dot(X, params)))

def RMSEfun(target, forecast):
    return np.sqrt(np.nanmean(np.power((target - forecast),2)))


