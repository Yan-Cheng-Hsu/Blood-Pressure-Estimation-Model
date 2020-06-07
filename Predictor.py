import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Model , Sequential
from keras.layers import Input, LSTM , Bidirectional , Dropout , Activation , Dense , Add , GRU , concatenate
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf
from keras.models import load_model , save_model
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ' 4 '

def rmse_loss (y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true - y_pred)))

def mae_loss (y_true, y_pred):
    return K.mean(K.abs(y_true - y_pred))
    
lambda_mse = 10 # hyperparameter to be adjusted

def joint_loss (y_true, y_pred):
    rmse_loss = K.sqrt(K.mean(K.square(y_true - y_pred)))
    mae_loss = K.mean(K.abs(y_true - y_pred))
    
    return mae_loss + (lambda_mse * rmse_loss)



print('feature loading......')
features = np.load( '/data1/Yan-Cheng-Hsu/TimeSeriesFeatures/TDFE_20200601/TDFE_20200601_32i_Features.npy' , allow_pickle = True )
bp = np.load( '/data1/Yan-Cheng-Hsu/TimeSeriesFeatures/TDFE_20200601/TDFE_20200601_32i_BP.npy' , allow_pickle = True )


print( 'feature loading finished.' )




print( 'Preprocessing......' )

RawFeatures = []
RawBP = []
Features = []
BP = []
maxList = []
minList = []
avgList = []

for x in range( len(bp) ):
    zerol = []
    for i in range( len(bp[x]) ):
        if bp[x][i][1] == 0:
            zerol.append( i )
            
    if len(zerol) != 0 :  
        
        if ( zerol[0] == 0 ) and ( zerol[ len(zerol)-1 ] == ( len(bp[x])-1 ) ) :
            for i in range( 1 , len(zerol) , 1 ):
                tempf = features[x][ (zerol[i-1]+1) : zerol[i] ]
                tempb = bp[x][ (zerol[i-1]+1) : zerol[i] ]
                if( len(tempb) != 0 ):
                    RawFeatures.append( tempf )
                    RawBP.append( tempb )
        elif ( zerol[0] == 0 ) and ( zerol[ len(zerol)-1 ] != ( len(bp[x])-1 ) ) :
            for i in range( 1 , len(zerol) , 1 ):
                tempf = features[x][ (zerol[i-1]+1) : zerol[i] ]
                tempb = bp[x][ (zerol[i-1]+1) : zerol[i] ]
                if( len(tempb) != 0 ):
                    RawFeatures.append( tempf )
                    RawBP.append( tempb )
            tempf = features[x][ (zerol[len(zerol)-1])+1 : len(features[x])-1 ]
            tempb = bp[x][ (zerol[len(zerol)-1])+1 : len(bp[x])-1 ]
            if( len(tempb) != 0 ):
                RawFeatures.append( tempf )
                RawBP.append( tempb )
        elif ( zerol[0] != 0 ) and ( zerol[ len(zerol)-1 ] == ( len(bp[x])-1 ) ) :
            tempf = features[x][ 0 : zerol[0] ]
            tempb = bp[x][ 0 : zerol[0] ]
            if( len(tempb) != 0 ) :
                RawFeatures.append( tempf )
                RawBP.append( tempb )
            for i in range( 1 , len(zerol) , 1 ):
                tempf = features[x][ (zerol[i-1]+1) : zerol[i] ]
                tempb = bp[x][ (zerol[i-1]+1) : zerol[i] ]
                if( len(tempb) != 0 ) :
                    RawFeatures.append( tempf )
                    RawBP.append( tempb )
        else :
            tempf = features[x][ 0 : zerol[0] ]
            tempb = bp[x][ 0 : zerol[0] ]
            if( len(tempb) != 0 ):
                RawFeatures.append( tempf )
                RawBP.append( tempb )
            for i in range( 1 , len(zerol) , 1 ):
                tempf = features[x][ (zerol[i-1]+1) : zerol[i] ]
                tempb = bp[x][ (zerol[i-1]+1) : zerol[i] ]
                if( len(tempb) != 0 ) :
                    RawFeatures.append( tempf )
                    RawBP.append( tempb )
            tempf = features[x][ (zerol[len(zerol)-1])+1 : len(features[x])-1 ]
            tempb = bp[x][ (zerol[len(zerol)-1])+1 : len(bp[x])-1 ]
            if( len(tempb) != 0 ) :
                RawFeatures.append( tempf )
                RawBP.append( tempb )     
    else :
        if( len(bp[x]) != 0 ):
            RawFeatures.append( features[x] )
            RawBP.append( bp[x] )



for i in range( len(RawFeatures) ):

    if len(RawFeatures[i]) < 1:
        continue
    else:
        for j in range( 1 , len(RawFeatures[i]) , 1 ):
            Features.append( RawFeatures[i][j-1:j] )
            BP.append( RawBP[i][j-1:j] )
Features = np.array( Features )
BP = np.array( BP )


for i in range( len(Features) ):
    maxList.append( np.max( Features[i] , axis = 0 ) )
    minList.append( np.min( Features[i] , axis = 0 ) )
    avgList.append( np.mean( Features[i] , axis = 0 ) )
    

maxList = np.array( maxList )
minList = np.array( minList )
avgList = np.array( avgList )
    
maxArray = np.max( maxList , axis = 0 )
minArray = np.min( minList , axis = 0 )
avgArray = np.mean( avgList , axis = 0 )

for i in range( len(Features) ):
    Features[i] = ( Features[i] - avgArray ) / ( maxArray - minArray )


x_train , x_test , y_train , y_test = train_test_split( Features , BP , test_size = 0.2 )


X_train = np.ndarray( ( len(x_train) , 1 , 32 ), dtype=float)
for i in range( len(x_train) ):
    X_train[i] = x_train[i]
X_test = np.ndarray( ( len(x_test) , 1 , 32 ), dtype=float)
for i in range( len(x_test) ):
    X_test[i] = x_test[i]
Y_train = np.ndarray( ( len(y_train) , 1 , 2 ), dtype=float)
for i in range( len(y_train) ):
    Y_train[i] = y_train[i]
Y_test = np.ndarray( ( len(y_test) , 1 , 2 ), dtype=float)
for i in range( len(y_test) ):
    Y_test[i] = y_test[i]




print( 'Preprocessing finished.' )



LayerUnits = 1024 * 2
Inputshape = ( len(Features[0]) , len(Features[0][0]) )
path = '/data1/Yan-Cheng-Hsu/TimeSeriesModel/TSM_20200601/TSM_20200601_4/TSM_20200601_4_1.h5'

#Initialize 
X_input = Input(Inputshape)

X0 = Dense(LayerUnits)(X_input)
X0 = Dropout(0.2)(X0)
X0 = Activation('relu')(X0)
X0 = Dense(LayerUnits*2)(X0)
X0 = Dropout(0.2)(X0)
X0 = Activation('relu')(X0)
X0 = Dense(LayerUnits*4)(X0)
X0 = Dropout(0.2)(X0)
X0 = Activation('relu')(X0)
X0 = Dense(LayerUnits)(X0)
X0 = Dropout(0.2)(X0)
X0 = Activation('relu')(X0)
X_output = Dense(2)(X0)

GRUModel = Model(inputs = X_input, outputs = X_output )


# Compiling
#GRUModel = multi_gpu_model( GRUModel , gpus = 3 )
GRUModel.compile(optimizer = 'Nadam' , loss=joint_loss, metrics=[rmse_loss, mae_loss] )
GRUModel.summary()

GRUModel.fit(X_train, Y_train, validation_split = 0.1 , epochs = 500 , batch_size = 512 )

print( 'saving model......' )
GRUModel.save( path )
print( 'finished model saving.' )

print( 'performance on training data : ' )
print( GRUModel.evaluate( X_train , Y_train ) )
print( 'performance on testing data : ' )
print( GRUModel.evaluate( X_test , Y_test ) )