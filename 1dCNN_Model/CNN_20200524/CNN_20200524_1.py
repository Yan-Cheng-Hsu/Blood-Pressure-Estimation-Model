import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model , Sequential
from keras.layers import Input, LSTM , Bidirectional , Dropout , Activation , Dense , Add , GRU
from keras.layers import Conv1D , MaxPooling1D , Flatten , AveragePooling1D , concatenate , BatchNormalization
from keras.optimizers import Adam
from keras import backend as K
from keras.utils import multi_gpu_model
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ' 0 '

def rmse_loss (y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true - y_pred)))


print( 'loading data....' )
ppg_part1 = np.load( '/data1/Yan-Cheng-Hsu/ppgData/dataperC/ppg_part1.npy' , allow_pickle = True )
ppg_part2 = np.load( '/data1/Yan-Cheng-Hsu/ppgData/dataperC/ppg_part2.npy' , allow_pickle = True )
ppg_part3 = np.load( '/data1/Yan-Cheng-Hsu/ppgData/dataperC/ppg_part3.npy' , allow_pickle = True )
sbp = np.load( '/data1/Yan-Cheng-Hsu/ppgData/dataperC/sbp.npy' , allow_pickle = True )
dbp = np.load( '/data1/Yan-Cheng-Hsu/ppgData/dataperC/dbp.npy' , allow_pickle = True )
print( 'loading data finished.' )


print('data preprocessing......')
ppg_min = 100
ppg_max = -100

for x in range( len(ppg_part1) ):
    for i in range( len(ppg_part1[x]) ):
        temp_min = np.min( ppg_part1[x][i] )
        temp_max = np.max( ppg_part1[x][i] )
        if temp_min < ppg_min:
            ppg_min = temp_min
        if temp_max > ppg_max:
            ppg_max = temp_max
for x in range( len(ppg_part2) ):
    for i in range( len(ppg_part2[x]) ):
        temp_min = np.min( ppg_part2[x][i] )
        temp_max = np.max( ppg_part2[x][i] )
        if temp_min < ppg_min:
            ppg_min = temp_min
        if temp_max > ppg_max:
            ppg_max = temp_max
for x in range( len(ppg_part3) ):
    for i in range( len(ppg_part3[x]) ):
        temp_min = np.min( ppg_part3[x][i] )
        temp_max = np.max( ppg_part3[x][i] )
        if temp_min < ppg_min:
            ppg_min = temp_min
        if temp_max > ppg_max:
            ppg_max = temp_max

            
ppg_part1 = ( ppg_part1 - ppg_min ) / ( ppg_max - ppg_min )
ppg_part2 = ( ppg_part2 - ppg_min ) / ( ppg_max - ppg_min )
ppg_part3 = ( ppg_part3 - ppg_min ) / ( ppg_max - ppg_min )


ppg = []
for x in range( len(ppg_part1) ):
    for i in range( len(ppg_part1[x]) ):
        ppg.append( ppg_part1[x][i] )
        
for x in range( len(ppg_part2) ):
    for i in range( len(ppg_part2[x]) ):
        ppg.append( ppg_part2[x][i] )
for x in range( len(ppg_part3) ):
    for i in range( len(ppg_part3[x]) ):
        ppg.append( ppg_part3[x][i] )
ppg = np.array( ppg )


bp = []

for x in range( len(sbp) ):
    for i in range( len(sbp[x]) ):
        temp = []
        temp.append( sbp[x][i] )
        temp.append( dbp[x][i] )
        temp = np.array( temp )
        bp.append( temp )
bp = np.array( bp )

endIndex = int( len(ppg)*0.8 )
ppg_train = ppg[0:endIndex]
ppg_test = ppg[endIndex:]
bp_train = bp[0:endIndex]
bp_test = bp[endIndex:]

PPG_train = ppg_train.reshape( len(ppg_train) , 300 , 1 ).astype( float )
PPG_test = ppg_test.reshape( len(ppg_test) , 300 , 1 ).astype( float )
BP_train = bp_train.reshape( len(bp_train) , 2 ).astype( float )
BP_test = bp_test.reshape( len(bp_test) , 2 ).astype( float )

print('data preprocessing finished.')





LayerUnits = 1024*2

#model1 start
Inputshape = ( len(PPG_train[0]) , len(PPG_train[0][0]) )
X_input1 = Input( Inputshape )
X_CNN1 = Conv1D( 64 , int(len( PPG_train[0] )/4) , border_mode = 'same' , activation = 'relu' )( X_input1 )
X_CNN1 = MaxPooling1D( pool_size = 2 )( X_CNN1 )
X_CNN1 = Conv1D( 64 , 4 , border_mode = 'same' , activation = 'relu' )( X_CNN1 ) 
X_CNN1 = X_CNN1 = MaxPooling1D( pool_size = 2 )( X_CNN1 )
X_CNN1 = Conv1D( 128 , 4 , border_mode = 'same' , activation = 'relu' )( X_CNN1 ) 
X_CNN1 = X_CNN1 = MaxPooling1D( pool_size = 2 )( X_CNN1 )
X_CNN1 = Dropout(0.25)(X_CNN1)
X_CNN1 = BatchNormalization()( X_CNN1 )
X_CNN1 = Flatten()(X_CNN1)


X = Dense(LayerUnits , activation = 'relu')(X_CNN1)
X = Dropout(0.2)(X)
X = Dense(LayerUnits*2 , activation = 'relu')(X)
X = Dropout(0.2)(X)
X = Dense(LayerUnits*4 , activation = 'relu')(X)
X = Dropout(0.2)(X)
X_output = Dense(2)(X)

GRUModel = Model( inputs = X_input1  , outputs = X_output )
# Compiling
GRUModel.compile(optimizer = 'adam', loss = rmse_loss )
GRUModel.summary()
GRUModel.fit( PPG_train , BP_train, validation_split = 0.1 , epochs = 500 , batch_size = 1024)

path = '/data1/Yan-Cheng-Hsu/CNNModel/CNN_20200524/CNN_20200524_1.h5'
print( 'saving model.....' )
GRUModel.save( path )
print( 'finish saving model.....' )



print( 'Training set : '  ,  GRUModel.evaluate( PPG_test , BP_test ) )
print( 'Testing set :' , GRUModel.evaluate( PPG_test , BP_test ) )

