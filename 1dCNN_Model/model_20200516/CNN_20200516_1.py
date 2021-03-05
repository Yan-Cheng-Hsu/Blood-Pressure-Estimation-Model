import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model , Sequential
from keras.layers import Input, LSTM , Bidirectional , Dropout , Activation , Dense , Add , GRU
from keras.layers import Conv1D , MaxPooling1D , Flatten , AveragePooling1D , concatenate
from keras.optimizers import Adam
from keras import backend as K
from keras.utils import multi_gpu_model
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ' 3 , 4 , 5 , 6'


def rmse_loss (y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true - y_pred)))

print( 'loading data....' )
ppg_part1 = np.load( '/data1/Yan-Cheng-Hsu/ppgData/dataperC/ppg_part1.npy' , allow_pickle = True )
ppg_part2 = np.load( '/data1/Yan-Cheng-Hsu/ppgData/dataperC/ppg_part2.npy' , allow_pickle = True )
ppg_part3 = np.load( '/data1/Yan-Cheng-Hsu/ppgData/dataperC/ppg_part3.npy' , allow_pickle = True )
diff1_part1 = np.load( '/data1/Yan-Cheng-Hsu/ppgData/dataperC/diff1_part1.npy' , allow_pickle = True )
diff1_part2 = np.load( '/data1/Yan-Cheng-Hsu/ppgData/dataperC/diff1_part2.npy' , allow_pickle = True )
diff1_part3 = np.load( '/data1/Yan-Cheng-Hsu/ppgData/dataperC/diff1_part3.npy' , allow_pickle = True )
diff2_part1 = np.load( '/data1/Yan-Cheng-Hsu/ppgData/dataperC/diff2_part1.npy' , allow_pickle = True )
diff2_part2 = np.load( '/data1/Yan-Cheng-Hsu/ppgData/dataperC/diff2_part2.npy' , allow_pickle = True )
diff2_part3 = np.load( '/data1/Yan-Cheng-Hsu/ppgData/dataperC/diff2_part3.npy' , allow_pickle = True )
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

diff1_min = 100
diff1_max = -100

for x in range( len(diff1_part1) ):
    for i in range( len(diff1_part1[x]) ):
        temp_min = np.min( diff1_part1[x][i] )
        temp_max = np.max( diff1_part1[x][i] )
        if temp_min < diff1_min:
            diff1_min = temp_min
        if temp_max > diff1_max:
            diff1_max = temp_max
for x in range( len(diff1_part2) ):
    for i in range( len(diff1_part2[x]) ):
        temp_min = np.min( diff1_part2[x][i] )
        temp_max = np.max( diff1_part2[x][i] )
        if temp_min < diff1_min:
            diff1_min = temp_min
        if temp_max > diff1_max:
            diff1_max = temp_max
for x in range( len(diff1_part3) ):
    for i in range( len(diff1_part3[x]) ):
        temp_min = np.min( diff1_part3[x][i] )
        temp_max = np.max( diff1_part3[x][i] )
        if temp_min < diff1_min:
            diff1_min = temp_min
        if temp_max > diff1_max:
            diff1_max = temp_max

diff2_min = 100
diff2_max = -100
for x in range( len(diff2_part1) ):
    for i in range( len(diff2_part1[x]) ):
        temp_min = np.min( diff2_part1[x][i] )
        temp_max = np.max( diff2_part1[x][i] )
        if temp_min < diff2_min:
            diff2_min = temp_min
        if temp_max > diff2_max:
            diff2_max = temp_max
for x in range( len(diff2_part2) ):
    for i in range( len(diff2_part2[x]) ):
        temp_min = np.min( diff2_part2[x][i] )
        temp_max = np.max( diff2_part2[x][i] )
        if temp_min < diff2_min:
            diff2_min = temp_min
        if temp_max > diff2_max:
            diff2_max = temp_max
for x in range( len(diff2_part3) ):
    for i in range( len(diff2_part3[x]) ):
        temp_min = np.min( diff2_part3[x][i] )
        temp_max = np.max( diff2_part3[x][i] )
        if temp_min < diff2_min:
            diff2_min = temp_min
        if temp_max > diff2_max:
            diff2_max = temp_max

            
ppg_part1 = ( ppg_part1 - ppg_min ) / ( ppg_max - ppg_min )
ppg_part2 = ( ppg_part2 - ppg_min ) / ( ppg_max - ppg_min )
ppg_part3 = ( ppg_part3 - ppg_min ) / ( ppg_max - ppg_min )
diff1_part1 = ( diff1_part1 - diff1_min ) / ( diff1_max - diff1_min )
diff1_part2 = ( diff1_part2 - diff1_min ) / ( diff1_max - diff1_min )
diff1_part3 = ( diff1_part3 - diff1_min ) / ( diff1_max - diff1_min )
diff2_part1 = ( diff2_part1 - diff2_min ) / ( diff2_max - diff2_min )
diff2_part2 = ( diff2_part2 - diff2_min ) / ( diff2_max - diff2_min )
diff2_part3 = ( diff2_part3 - diff2_min ) / ( diff2_max - diff2_min )






ppg = []
diff1 = []
diff2 = []
for x in range( len(ppg_part1) ):
    if len(ppg_part1[x]) >= 10:
        for i in range( 10 , len(ppg_part1[x]) , 10 ):
            ppg.append( ppg_part1[x][i-10:i] )
for x in range( len(ppg_part2) ):
    if len(ppg_part2[x]) >= 10:
        for i in range( 10 , len(ppg_part2[x]) , 10 ):
            ppg.append( ppg_part2[x][i-10:i] )
for x in range( len(ppg_part3) ):
    if len(ppg_part3[x]) >= 10:
        for i in range( 10 , len(ppg_part3[x]) , 10 ):
            ppg.append( ppg_part3[x][i-10:i] )

for x in range( len(diff1_part1) ):
    if len(diff1_part1[x]) >= 10:
        for i in range( 10 , len(diff1_part1[x]) , 10 ):
            diff1.append( diff1_part1[x][i-10:i] )
for x in range( len(diff1_part2) ):
    if len(diff1_part2[x]) >= 10:
        for i in range( 10 , len(diff1_part2[x]) , 10 ):
            diff1.append( diff1_part2[x][i-10:i] )
for x in range( len(diff1_part3) ):
    if len(diff1_part3[x]) >= 10:
        for i in range( 10 , len(diff1_part3[x]) , 10 ):
            diff1.append( diff1_part3[x][i-10:i] )
            

for x in range( len(diff2_part1) ):
    if len(diff2_part1[x]) >= 10:
        for i in range( 10 , len(diff2_part1[x]) , 10 ):
            diff2.append( diff2_part1[x][i-10:i] )
for x in range( len(diff2_part2) ):
    if len(diff2_part2[x]) >= 10:
        for i in range( 10 , len(diff2_part2[x]) , 10 ):
            diff2.append( diff2_part2[x][i-10:i] )
for x in range( len(diff2_part3) ):
    if len(diff2_part3[x]) >= 10:
        for i in range( 10 , len(diff2_part3[x]) , 10 ):
            diff2.append( diff2_part3[x][i-10:i] )




bp = []
for x in range( len(sbp) ):
    if len(sbp) >= 10:
        for i in range( 10 , len(sbp[x]) , 10 ):
            temp_s = sbp[x][i-10:i]
            temp_d = dbp[x][i-10:i]
            bp_list = []
            for j in range( len(temp_s) ):
                temp_bp = []
                temp_bp.append( temp_s[j] )
                temp_bp.append( temp_d[j] )
                bp_list.append( temp_bp )
            bp_list = np.array( bp_list )
            bp.append( bp_list )            



ppg_part1 = []
ppg_part2 = []
ppg_part3 = []
diff1_part1 = []
diff1_part2 = []
diff1_part3 = []
diff2_part1 = []
diff2_part2 = []
diff2_part3 = []
sbp = []
dbp = []


endIndex = int( len(ppg)*0.8 )
ppg_train = ppg[0:endIndex]
ppg_test = ppg[endIndex:]
diff1_train = diff1[0:endIndex]
diff1_test = diff1[endIndex:]
diff2_train = diff2[0:endIndex]
diff2_test = diff2[endIndex:]
bp_train = bp[0:endIndex]
bp_test = bp[endIndex:]



PPG_train = np.ndarray( ( len(ppg_train) , len(ppg_train[0]) , len(ppg_train[0][0]) ), dtype=float)
for i in range( len(ppg_train) ):
    PPG_train[i] = ppg_train[i]
PPG_test = np.ndarray( ( len(ppg_test) , len(ppg_test[0]) , len(ppg_test[0][0]) ), dtype=float)
for i in range( len(ppg_test) ):
    PPG_test[i] = ppg_test[i]
DIFF1_train = np.ndarray( ( len(diff1_train) , len(diff1_train[0]) , len(diff1_train[0][0]) ), dtype=float)
for i in range( len(diff1_train) ):
    DIFF1_train[i] = diff1_train[i] 
DIFF1_test = np.ndarray( ( len(diff1_test) , len(diff1_test[0]) , len(diff1_test[0][0]) ), dtype=float)
for i in range( len(diff1_test) ):
    DIFF1_test[i] = diff1_test[i] 
DIFF2_train = np.ndarray( ( len(diff2_train) , len(diff2_train[0]) , len(diff2_train[0][0]) ), dtype=float)
for i in range( len(diff2_train) ):
    DIFF2_train[i] = diff2_train[i] 
DIFF2_test = np.ndarray( ( len(diff2_test) , len(diff2_test[0]) , len(diff2_test[0][0]) ), dtype=float)
for i in range( len(diff2_test) ):
    DIFF2_test[i] = diff2_test[i] 

BP_train = np.ndarray( ( len(bp_train) , len(bp_train[0]) , len(bp_train[0][0]) ), dtype=float)
for i in range( len(bp_train) ):
    BP_train[i] = bp_train[i]
BP_test = np.ndarray( ( len(bp_test) , len(bp_test[0]) , len(bp_test[0][0]) ), dtype=float)
for i in range( len(bp_test) ):
    BP_test[i] = bp_test[i]

print('data preprocessing finished.')




#model1 start
Inputshape = ( len(PPG_train[0]) , len(PPG_train[0][0]) )
X_input = Input( Inputshape )
X_CNN = Conv1D( 64 , 3 , border_mode = 'same' , activation = 'relu' )( X_input )
X_CNN = Dropout( 0.25 )( X_CNN )
X_CNN = Conv1D( 64 , 3 , border_mode = 'same' , activation = 'relu' )( X_CNN )
X_CNN = Dropout( 0.25 )( X_CNN )
X_CNN = Conv1D( 128 , 3 , border_mode = 'same' , activation = 'relu' )( X_CNN )
X_CNN = Dropout( 0.25 )( X_CNN )
X_CNN = Conv1D( 128 , 3 , border_mode = 'same' , activation = 'relu' )( X_CNN )
X_CNN = Dropout( 0.25 )( X_CNN )

LayerUnits = 256
X0 = Bidirectional(GRU(LayerUnits, return_sequences=True), merge_mode='concat')(X_CNN)
X0 = Dropout(0.2)(X0)
X0 = Activation('relu')(X0)

X = GRU(LayerUnits, return_sequences=True)(X0)
X = Dropout(0.2)(X)
X = Activation('relu')(X)

X1 = GRU(LayerUnits, return_sequences=True)(X)
X1 = Dropout(0.2)(X1)
X1 = Activation('relu')(X1)

#X2 = concatenate([X, X1])
X2 = GRU(LayerUnits, return_sequences=True)(X1)
X2 = Dropout(0.2)(X2)
X2 = Activation('relu')(X2)


#X3 = concatenate( [X2,X] )
X3 = GRU(LayerUnits, return_sequences=True)(X2)
X3 = Dropout(0.2)(X3)
X3 = Activation('relu')(X3)

X3 = GRU(LayerUnits, return_sequences=True)(X3)
X3 = Dropout(0.2)(X3)
X3 = Activation('relu')(X3)


X_output = Dense(2)(X3)

GRUModel = Model(inputs =X_input, outputs = X_output )



max_epoch = 100
init_lr = 0.003 
opt = Adam(lr=init_lr, decay=init_lr / max_epoch)

# Compiling
GRUModel.compile(optimizer = opt , loss = rmse_loss )
GRUModel.summary()
GRUModel.fit(PPG_train, BP_train, validation_split = 0.1 , epochs = 100 , batch_size = 16)

path = '/data1/Yan-Cheng-Hsu/CNNModel/model_20200516/CNN_model_20200516_1.h5'
print( 'saving model.....' )
GRUModel.save( path )
print( 'finish saving model.....' )



print( 'Training set : '  ,  GRUModel.evaluate( PPG_test , BP_test ) )
print( 'Testing set :' , GRUModel.evaluate( PPG_test , BP_test ) )

