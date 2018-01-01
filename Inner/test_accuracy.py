import numpy as np



path = 'train/default_model_CV_{}/'


accs = []
mse = []
rmse = []
aae = []
for i in range(10):
    
    if 1:
        pred = np.load(path.format(i)+'predictions_valid.npy')
        targets = np.load(path.format(i)+'targets_valid.npy')
    else:
        pred = np.load(path.format(i)+'predictions_train.npy')
        targets = np.load(path.format(i)+'targets_train.npy')
    
    print pred.shape, targets.shape, pred.dtype, targets.dtype
    mse.append(np.mean((pred-targets)**2))
    aae.append(np.mean(np.abs(pred-targets)))
    rmse.append(np.sqrt(np.mean((pred-targets)**2)))
    print 'mse, rmse',mse[-1], rmse[-1]
    accs.append(np.mean(((pred>0.5).astype(bool)==(targets>0.5).astype(bool)))*100 )
    print accs[-1]
    
    
print 'average accuracy:', np.mean(accs)
print 'average mse:', np.mean(mse)
print 'average rmse:', np.mean(rmse)
print 'average AAE:', np.mean(aae)





