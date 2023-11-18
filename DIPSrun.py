import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import load
from numpy import save
from numpy import asarray

import h5py
varnames=['sd0','sz0','st','ptfrac','dr','numPix','numSCT','d0','z0']

dname = f'/eos/home-c/cmauceri/DIPS_acts/DIPS-4d-tracking/data/output200PU.hdf5'
f = h5py.File(dname,"r")

def remIV(X_train,X_test,idx):
    selected_train=np.delete(X_train,idx,axis=2)
    selected_test=np.delete(X_test,idx,axis=2)
    return selected_train,selected_test

def DIPSrun(f,X_train,X_test,nincvar):
    y_train       = f['y_train'][:]
    ix_train      = f['ix_train'][:]
    weights_train = f['weights_train'][:] # Weights to avoid learning directly from the pT dist
    
    nClasses = 3 # Only consider l, c, and b-jets

    if y_train.max() >= nClasses:

        valid_targets = (y_train < nClasses)

        X_train       = X_train[valid_targets]
        y_train       = y_train[valid_targets]
        ix_train      = ix_train[valid_targets]
        weights_train = weights_train[valid_targets]
        
    #Transform the input vector to a one-hot vector
    y_train[:4]
    from keras.utils.np_utils import to_categorical

    y_train_cat = to_categorical(y_train, num_classes=nClasses)
    
    # Setting up architecture
    from tensorflow import keras
    from keras.layers import BatchNormalization, Layer, TimeDistributed
    from keras.layers import Dense, Input, ReLU, Masking
    from keras.models import Model

    from keras.callbacks import EarlyStopping, ModelCheckpoint
    from keras import backend as K
    
    class Sum(Layer):
        """
        Simple sum layer à la Dan Guest
        """

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.supports_masking = True

        def build(self, input_shape):
            pass

        def call(self, x, mask=None):
            if mask is not None:
                x = x * K.cast(mask, K.dtype(x))[:,:,None]
            return K.sum(x, axis=1)

        def compute_output_shape(self, input_shape):
            return input_shape[0], input_shape[2]

        def compute_mask(self, inputs, mask):
            return None
    
    # Store the dimensions of the input dataset for setting up the model
    nJets, maxNumTrks, nFeatures = X_train.shape
    ppm_sizes_int = [50,50,32]
    dense_sizes_int = [50,50]
    
    trk_inputs = Input(shape=(maxNumTrks,nFeatures),name='Input')
    masked_inputs = Masking(mask_value=0,name='Mask')(trk_inputs)
    tdd = masked_inputs

    for i, phi_nodes in enumerate(ppm_sizes_int):

        tdd = TimeDistributed(Dense(phi_nodes,activation='linear'),name=f"Phi{i}_Dense")(tdd)
        tdd = TimeDistributed(ReLU(),name=f"Phi{i}_ReLU")(tdd)

    # This is where the magic happens... sum up the track features!
    F = Sum(name="Sum")(tdd)

    for j, F_nodes in enumerate(dense_sizes_int):

        F = Dense(F_nodes, activation='linear', name=f"F{j}_Dense")(F)
        F = ReLU(name=f"F{j}_ReLU")(F)

    output = Dense(nClasses, activation='softmax',name="Jet_class")(F)
    dips = Model(inputs=trk_inputs, outputs=output)
    opt = keras.optimizers.Adam(learning_rate=0.0001)

    dips.compile(loss='categorical_crossentropy', optimizer='adam',
             metrics=['acc'])  
    dips.evaluate(X_train,y_train_cat,batch_size=128)
    nEpochs = 100

    earlyStop = EarlyStopping(monitor='val_loss', verbose=True, patience=10)

    dips_mChkPt = ModelCheckpoint('dips_weights_no_'+nincvar+'.h5',
                              monitor='val_loss', 
                              verbose=True,
                              save_best_only=True,
                              save_weights_only=True)
    dips_hist = dips.fit(X_train, y_train_cat, epochs=nEpochs, 
                     batch_size=128,validation_split=0.2,
                     callbacks=[earlyStop, dips_mChkPt])
    epochs = np.arange(1,len(dips_hist.history['loss'])+1)

    
    y_test = f['y_test'][:]
    title = 'DIPS: $\Phi$ ' + '-'.join([str(i) for i in ppm_sizes_int])
    title += ', F ' + '-'.join([str(i) for i in dense_sizes_int])
    
    effs = sigBkgEff(dips, X_test, y_test, nincvar,title=title)
    return effs
def sigBkgEff(myModel, X_test, y_test, nincvar,fc=0.07, title=''):
    '''
    Given a model get the ROC curves.
    Input:
        myModel: A keras model
        X_test: Model inputs of the test set
        y_test: Truth labels for the test set
        fc: The amount by which to weight the c-jet prob in the disc. The
            default value of 0.07 corresponds to the fraction of c-jet bkg
            in ttbar.

    Output:
        effs: A list with 3 entries for the l, c, and b effs
    '''

    # Evaluate the performance with the ROC curves!
    predictions = myModel.predict(X_test,verbose=True)

    '''
    TO DO: Set up the discriminant function
    '''
    disc = np.log(np.divide((predictions[:,2]),(fc*predictions[:,1]+(1-fc)*predictions[:,0])))
    
    # Define the min and max range for the distribution
    real = (disc != -np.inf) & (disc != np.inf)
    np.min(disc[real]),np.max(disc[real])
    
    discMin = np.min(disc[real])
    discMax = np.max(disc[real])
    print("Db discriminant range:",discMin,discMax)
    
    # Reset the -inf and +inf to have ranges at the edge of the distribution
    eps=1e-10
    disc[disc == -np.inf] = discMin+eps
    disc[disc ==  np.inf] = discMax-eps
   
    myRange=(discMin,discMax)
    nBins = 200

    effs = []
    for i, flavor in zip([0,1,2], ['l','c','b']):

        ix = (y_test == i)
        
        # Plot the discriminant output
        # nEntries is just a sum of the weight of each bin in the histogram.
        hi = plt.hist(disc[ix],nBins, myRange, histtype='step', density=True,
                      label=f'{flavor}-jets', log=True)[0]
        
        # Since high Db scores correspond to more b-like jets, compute the cummulative density function
        # from summing from high to low values, this is why we reverse the order of the bins in nEntries
        # using the "::-1" numpy indexing.
        eff = np.add.accumulate(hi[::-1]) / np.sum(hi)
        effs.append(eff)
    return effs
X_train=f['X_train'][:]
X_test=f['X_test'][:]
full_eff=DIPSrun(f,X_train,X_test,'')
full_effArray=asarray(full_eff)
save('roc_data/allvar.npy',full_effArray)
for j in range(len(varnames)):
    nwX_train=remIV(X_train,X_test,j)[0]
    nwX_test=remIV(X_train,X_test,j)[1]
    nincvar=varnames[j]
    print(np.shape(nwX_train))
    eff=DIPSrun(f,nwX_train,nwX_test,nincvar)
    effArray=asarray(eff)
    save('roc_data/no_'+str(nincvar)+'.npy',effArray)
#selected_train=np.delete(X_train,6,axis=2)
#selected_test=np.delete(X_test,6,axis=2)
#selected_train=np.delete(selected_train,0,axis=2)
#selected_test=np.delete(selected_test,0,axis=2)
#eff=DIPSrun(f,selected_train,selected_test,'sd0_d0')
#effArray=asarray(eff)
#save('roc_data/no_sd0_d0.npy',effArray)
    
