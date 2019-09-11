IN USER_PARAMETERS EVERY ENSEMBLE MODEL NEED AN ELEMENT CALLED "aggregationModel", WHICI IS A DICTIONARY WITH TWO FIELDS. 

THE FIRST ONE IS "name", WHOSE VALUE MUST BE THE NAME OF THE AGGREGATION METHOD (INSIDE THE SCRIPT "ensembleAggregator.py"), IN STRING FORMAT, THAT WE WANT TO USE FOR OUR ENSEMBLE MODEL.
THE SECOND IS "characteristics", WHOSE VALUE MUST BE A DICTIONARY WITH A MANDATORY ELEMENT CALLED "baseLearners", WHOSE VALUE MUST BE A LIST OF THE MODELS NAMES (STRING FORMAT), THAT WE WANT TO COMBINE.

ALL THE OTHER POTENTIAL ELEMENTS ON THE DICTIONARY "characteristics", DEPEND ON THE AGGREGATION TECHNIQUE CONSIDERED AND THEY ARE LISTED BELOW.

N.B.: EACH ELEMENT FOR WHICH THERE IS NO DEFAULT SPECIFIED IS MANDATORY


##########################

SIMPLE AVERAGE:

name : 'simple_average'
characteristics : {"baseLearners" : baseLearners_list
                   "handlesMissing" : boolean,                          # wheter the considered method is able to deal with missing features, DEFAULT "False"}


##########################

TRIMMED AVERAGE:

name : 'trimmed_average'
characteristics : {"baseLearners" : baseLearners_list}


##########################

WINDSORIZED AVERAGE

name : 'windsorized_average'
characteristics : {"baseLearners" : baseLearners_list}


##########################

ORDINARY LEAST SQUARES AVERAGE

name : 'ols_average'
characteristics : {"baseLearners" : baseLearners_list,
                   "bias" : boolean}                                   # wheter to consider a constant term or not in the model, DEFAULT "False"


###########################

LEAST ABSOLUTE DEVIATION AVERAGE

name : 'lad_average'
characteristics : {"baseLearners" : baseLearners_list,
                   "bias" : boolean}                                   # wheter to consider a constant term or not in the model, DEFAULT "False"


###########################

POSITIVE WEIGHTS AVERAGE

name : 'pw_average'
characteristics : {"baseLearners" : baseLearners_list}


###########################

CONSTRAINED LEAST SQUARES AVERAGE

name : 'cls_average'
characteristics : {"baseLearners" : baseLearners_list}


###########################

MULTILAYER PERCEPTRON REGRESSION

name : 'mlp_regression'
characteristics : {"baseLearners" : baseLearners_list,
                   "activation" : string,                               # name of the activation function: possible choices are: 'identity', 'relu', 'logistic', 'tanh'
                   "alpha" : float,                                     # L2 penalty (regularization term) parameter
                   "hidden_layer_sizes" : tuple}                        # tuple of integer numbers, where the ith element represent the number of neurons in the ith hidden layer


###########################

INVERSE ROOT MEAN SQUARED ERROR AVERAGE

name : 'irmse_average'
characteristics : {"baseLearners" : baseLearners_list,
                   "trainWindowHours" : integer}                        # integer that specify the length (in hours) of the recursive training window
                   

###########################

SEQUENTIAL CONVEX AGGREGATION METHOD WITH FIXED-SHARE RULE

name : 'sc_fixed_share_aggregation'
characteristics : {"baseLearners" : baseLearners_list,
                   "handlesMissing" : boolean,                          # wheter the considered method is able to deal with missing features, DEFAULT "False"
                   "trainWindowHours" : integer                         # integer that specify the length (in hours) of the recursive training window
                   "initialization" : {                                 # dictionary for the initialization of the parameters of the method
                                       "eta" : float,                   # initialization of the exponential parameter. It must be > 0
                                       "alpha" : float,                 # initialization of the mixture parameter. It must be >= 0 and <= 1
                                       "weights" : list                 # list of integers representing the initialization of the weights vector. It must be convex (that is, each element must be >= 0 
                                      }                                 # and the sum of all the elements must be 1)


                                      
        
