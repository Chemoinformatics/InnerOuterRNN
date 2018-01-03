from __future__ import print_function

import time
import numpy as np
import sklearn.metrics as metrics


from matplotlib import pyplot
import Keras.backend as backend

from . import utils



def lim(float, precision = 5):
    return ("{0:."+str(precision)+"f}").format(float)



def save_model_visualization(model, filename='model.png'):
    '''
    Requires the 'graphviz' software package
    '''
    try:
        from Keras.utils.visualize_util import plot
        plot(model, filename, show_shapes=1)
    except:
        import traceback
        print('\nsave_model_visualization() failed with exception:',traceback.format_exc())



def predict(data, model):
    '''
    Returns a tensor containing the DNN's predictions for the given list of batches <data>
    '''
    pred = []
    for batch in data:
        if len(batch)==2:
            batch = batch[0]
        pred.append(model.predict_on_batch(batch))
    return np.concatenate(pred)



def eval_metrics_on(predictions, labels, regression = True, Targets_UnNormalization_fn = lambda x:x):
    '''
    assuming this is a regression task; labels are continuous-valued floats

    returns most regression-related scores for the given predictions/targets as a dictionary:

        r2, mae, mse, rmse, median_absolute_error, explained_variance_score
    '''
    if len(labels[0])==2: #labels is list of data/labels pairs
        labels = np.concatenate([l[1] for l in labels])

    predictions = Targets_UnNormalization_fn(predictions)
    labels      = Targets_UnNormalization_fn(labels)

    if regression:
        predictions = predictions[:,0]
        r2                       = metrics.r2_score(labels, predictions)
        mae                      = np.abs(predictions - labels).mean()
        mse                      = ((predictions - labels)**2).mean()
        rmse                     = np.sqrt(mse)
        median_absolute_error    = metrics.median_absolute_error(labels, predictions) # robust to outliers
        explained_variance_score = metrics.explained_variance_score(labels, predictions) # best score = 1, lower is worse
        return {'r2':r2, 'mae':mae, 'mse':mse, 'rmse':rmse,
                'median_absolute_error':median_absolute_error, 'explained_variance_score':explained_variance_score, 'main_metric':rmse}
    else:
        if labels.shape[1] == 2*predictions.shape[1]:
            labels = labels.reshape((labels.shape[0], labels.shape[1]//2, 2))
            labels, mask = labels[:,:,0], labels[:,:,1]
        else:
            mask = None
            assert labels.shape == predictions.shape, 'Something seems wrong. Are all function arguments for the model and data-preprocessing correct?'

        if abs(np.mean(predictions.sum(1))-1) < 1e-4:
            #regular softmax predictions
            predictions = predictions[:,1]
            if labels.max()==1:
                fpr, tpr, thresholds = metrics.roc_curve(labels[:,1], predictions, pos_label=1)
                auc = metrics.auc(fpr, tpr, reorder=1)
            accuracy = np.mean((predictions>0.5)==labels[:,1])
        else:
            # multi-task binary
#            print('predictions.shape, labels.shape',predictions.shape, labels.shape)
#            print(predictions[:2])
            accuracy = np.mean((predictions>0.5)==labels)
            AUCs=[]
            for i in range(labels.shape[1]):
                fpr, tpr, thresholds = metrics.roc_curve(labels[:,i], predictions[:,i], pos_label=1, sample_weight=mask[:,i])
                auc = metrics.auc(fpr, tpr, reorder=1)
                AUCs.append(auc)
            auc = np.mean(AUCs)
            if np.isnan(auc):
                AUCs = np.asarray(AUCs)
                num_nan = np.sum(np.isnan(AUCs))
                auc = np.mean(np.nan_to_num(AUCs)) * len(AUCs)/(1.*len(AUCs)-num_nan) #mean ignoring NaN entries in AUCs

        return {'auc':auc, 'accuracy':accuracy, 'main_metric':accuracy}


def parity_plot(predictions, labels):
    try:
        figure = pyplot.figure()
    except:
        print('parity_plot:: Error: Cannot create figure')
        return
    ax  = figure.add_subplot(111)
    ax.set_axisbelow(True)

    ax.set_xlabel('True Value', fontsize=15)
    ax.set_ylabel('Predicted', fontsize=15)
    pyplot.grid(b=True, which='major', color='lightgray', linestyle='--')
    pyplot.title('Parity Plot')
    pyplot.scatter(labels, predictions, s=15, c='b', marker='o')




def test_on(data, model, description='test_data score:'):
    '''
    Returns the model's mse on the given data
    '''
    scores=[]
    weights =[]
    for v in data: # data, labels = v
        weights.append(len(v[1])) # size of batch
        scores.append( model.test_on_batch(x=v[0], y=v[1]))
    weights = np.array(weights)
    s=np.mean(np.array(scores)* weights/weights.mean())
    if len(description):
        print(description, lim(s))
    return s



def normalize_RegressionTargets(targets):
    '''
    returns normalized targets and the function to undo the normalization set (important for making actual predictions and for evaluating the model correctly)
    '''
    labels_mean  = targets.mean()
    labels_range = np.max(targets) - np.min(targets)
    targets = (targets - labels_mean)/labels_range

    #this function MUST be applied to predictions of the model and the targets when computing metrics
    def Targets_UnNormalization_fn(targets_):
        return targets_*labels_range + labels_mean
    return targets, Targets_UnNormalization_fn



def get_model_params(model):
    weight_values = []
    for lay in model.layers:
        weight_values.extend( backend.batch_get_value(lay.weights))
    return weight_values



def set_model_params(model, weight_values):
    symb_weights = []
    for lay in model.layers:
        symb_weights.extend(lay.weights)
    assert len(symb_weights) == len(weight_values)
    for model_w, w in zip(symb_weights, weight_values):
        backend.set_value(model_w, w)



def save_model_weights(model, filename = 'fingerprint_model_weights.npy'):
    ws = get_model_params(model)
    utils.save_npy(filename, ws)



def load_model_weights(model, filename = 'fingerprint_model_weights.npy'):
    ws = np.load(filename)
    set_model_params(model, ws[ws.keys()[0]])






def load_and_cache_csv(data_set = None, csv_file_name = None,
                       input_field_name = 'smiles', target_field_name = 'solubility',
                       cache_location = 'data/cached/'):
    '''
        Loads a csv-based data-set and caches the data afer processing.

        data_set (str or None) [optional]:
        --------------

            Specify the name of the cache and an example data set
            valid values: 'delaney', 'huuskonsen', 'bloodbrainbarrier', None

        Returns:
        ---------

            data, labels, regression, num_classes
    '''

    regression = True
    num_classes = 1

    if data_set == 'delaney':
        csv_file_name     = 'data/delaney.csv'
        input_field_name  = 'smiles'
        target_field_name ='measured log solubility in mols per litre'
    elif data_set == 'huuskonsen':
        csv_file_name = 'data/huuskonsen.csv'
        input_field_name  = 'smiles'
        target_field_name ='solubility'
    elif data_set == 'karthikeyan':
        csv_file_name = 'data/Melting_Points_(Karthikeyan).txt'
        input_field_name  = 'SMILES'
        target_field_name ='MTP'
    elif data_set == 'bloodbrainbarrier':
        csv_file_name = 'data/bbbp2__blood_brain_barrier_penetration_classification.csv'
        input_field_name  = 'smiles'
        target_field_name ='binary_penetration'
        loader_fn = utils.load_csv_file_wrapper(file = 'data/bbbp2__blood_brain_barrier_penetration_classification.csv', target_name='binary_penetration')
        regression = False
        num_classes = 2
    else:
        #raise ValueError('The specified data set is not recognized: '.fromat(data_set))
        assert isinstance(csv_file_name, str), 'Argument csv_file_name must be specified if data_set is not one of the valid default values'
        data_set = csv_file_name.replace('\\','/').split('/')[-1]

    loader_fn  = utils.load_csv_file_wrapper(file = csv_file_name, input_name = input_field_name, target_name=target_field_name)

    # load and preprocess; uses cached data if available
    if cache_location is None:
        cache_name = None
    else:
        if cache_location[-1] not in ['/','\\']:
            cache_location = cache_location+'/'
        cache_name = cache_location+data_set
    data, labels = utils.filter_data(loader_fn, data_cache_name = cache_name)

    print('Number of valid examples in data set:',len(data))

    if isinstance(target_field_name, list):
        assert len(target_field_name) == labels.shape[1], 'ERROR: Failed to correctly load the data'

    return data, labels, regression, num_classes



def create_labels_NaN_mask(labels):
    '''
    replaces NaN's with 0 in labels as well.
    attaches mask into labels array as separate channel - will be picked up by masked loss function.
    '''
    mask = np.where(np.isnan(labels), 0, 1).astype('int16')
    if not np.any(mask==0):
        return labels, False
    labels = np.nan_to_num(labels)
    labels = labels.astype('int16')
    labels = np.concatenate([labels[:,:,None],mask[:,:,None]], axis=2).reshape((labels.shape[0], -1))
    print('Missing entries in labels detected: masked loss will be used. (masked labels shape: {})'.format(labels.shape))
    return labels, True









def update_lr(model, initial_lr, relative_progress, total_lr_decay):
    """
    exponential decay

    initial_lr: any float (most reasonable values are in the range of 1e-5 to 1)
    total_lr_decay: value in (0, 1] -- this is the relative final LR at the end of training
    relative_progress: value in [0, 1] -- current position in training, where 0 == beginning, 1==end of training and a linear interpolation in-between
    """
    assert total_lr_decay > 0 and total_lr_decay <= 1
    backend.set_value(model.optimizer.lr, initial_lr * total_lr_decay**(relative_progress))







def plot_training_mse_evolution(data_lists, legend_names=[], ylabel = 'MSE', xlabel = 'training epoch', legend_location='best'):

    _colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    try:
        figure = pyplot.figure()
    except:
        print('plot_training_mse_evolution:: Error: Cannot create figure')
        return
    ax  = figure.add_subplot(111)
    ax.set_axisbelow(True)
    if len(legend_names):
        assert len(legend_names)==len(data_lists), 'you did not provide enough or too many labels for the graph'
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    pyplot.grid(b=True, which='major', color='lightgray', linestyle='--')
    if len(legend_names) != len(data_lists):
        legend_names = [' ' for x in data_lists]
    for i, data in enumerate(data_lists):
        assert len(data)==len(data_lists[0])
        pyplot.plot(np.arange(1,len(data)+1), data,
                    _colors[i%len(_colors)], linestyle='-', marker='o',
                    markersize=5, markeredgewidth=0.5, linewidth=2.5, label=legend_names[i])
    if len(legend_names[0]):
        ax.legend(loc=legend_location, shadow=0, prop={'size':14}, numpoints=1)



def dict_as_txt(dct):
    '''returns content of a dictionary formatted as multiline string'''
    return '\n'.join(['{:25s} {:6.5f}'.format(a.upper()+':',b) for a,b in dct.items()])



