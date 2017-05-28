import numpy as np

array = np.load('./Tyler/bottleneck_features_train.npy')
print ('Tyler')
print (np.shape(array))

flatten = array.reshape((array.shape[0], 4*4*512))

# 36 + 54 + 42 = 132
np.save('./Tyler/features_tyler_1.npy', flatten[0:36])
np.save('./Tyler/features_tyler_2.npy', flatten[36:36+54])
np.save('./Tyler/features_tyler_3.npy', flatten[36+54:36+54+42])
