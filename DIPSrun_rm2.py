import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import load
from numpy import save
from numpy import asarray
import h5py
filenames=[]
varnames=['sd0','d0','d0_sd0']
fullFile='data/output.hdf5_rm2no_none'
for var in varnames:
    filenames.append('data/output.hdf5_rm2no_'+var)

def DIPSrun(f,nincvar):
    print(f)
    X_train       = f['X_train'][:]
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


    # Making roc curve
    X_test = f['X_test'][:]
    y_test = f['y_test'][:]
    title = 'DIPS: $\Phi$ ' + '-'.join([str(i) for i in ppm_sizes_int])
    title += ', F ' + '-'.join([str(i) for i in dense_sizes_int])
    
    effs = sigBkgEff(dips, X_test, y_test, nincvar,title=title)
    return effs
def sigBkgEff(myModel, X_test, y_test, nincvar,fc=0.07, title=''):
    '''
    Given a model, make the histograms of the model outputs to get the ROC curves.

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
        hi = plt.hist(disc[ix],nBins, myRange, alpha=0.5, density=True,
                      label=f'{flavor}-jets', log=True)[0]
        
        # Since high Db scores correspond to more b-like jets, compute the cummulative density function
        # from summing from high to low values, this is why we reverse the order of the bins in nEntries
        # using the "::-1" numpy indexing.
        eff = np.add.accumulate(hi[::-1]) / np.sum(hi)
        effs.append(eff)

#    plt.title(title)
 #   plt.legend()
  #  plt.xlabel('$D = \ln [ p_b / (f_c p_c + (1- f_c)p_l ) ]$',fontsize=14)
   # plt.ylabel('Normalized entries')
   # plt.savefig("roc_no_"+nincvar+".png")    
    #plt.clf()
    return effs
parameterlist=[r'$d_{0}/\sigma_{d0}$',r'$d_{0}$ [mm]',r'$d_{0}/\sigma_{d0}$ or $d_{0}$ [mm]']
full_f= h5py.File(fullFile,"r")
feff=DIPSrun(full_f,'')
feffArray=asarray(feff)
save('roc_data/allvar.npy',feffArray)
for inputvar in range(len(varnames)):
    dname=filenames[inputvar]
    f = h5py.File(dname,"r")
    nincvar=varnames[inputvar]
    label = parameterlist[inputvar]
    eff= DIPSrun(f,nincvar)
    effArray=asarray(eff)
    save('roc_data/no_'+str(nincvar)+'.npy',effArray)
##    plt.figure()
  #  plt.plot()
     # l-rej
   # plt.figure()
    #plt.plot(n_beff, 1 / n_leff, color='hotpink', label='l-rej (no '+str(label))

    # c-rej
   # plt.plot(n_beff, 1 / n_ceff, color='hotpink', linestyle='--', label='c-rej (no '+str(label))
  #  plt.xlabel('b efficiency')
    #plt.ylabel('Background rejection')
   # l-rej
   # plt.plot(f_beff, 1 / f_leff, color='orange', label='l-rej')
    # c-rej
    #plt.plot(f_beff, 1 / f_ceff, color='orange', linestyle='--', label='c-rej')
  #  plt.xlabel('b efficiency')
   # plt.ylabel('Background rejection')
#    plt.title("ROC without "+str(label)+" vs. full Inputs")
 #   plt.legend()
  #  plt.yscale("log")
   # plt.xlim(0.6,1)
   # plt.savefig("roc_no_"+str(nincvar+".png"))
   # plt.clf()
