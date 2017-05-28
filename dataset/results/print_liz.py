import numpy as np

array = np.load('./Liz/bottleneck_features_train.npy')
print ('Liz')
print (np.shape(array))

flatten = array.reshape((array.shape[0], 4*4*512))

# 79 + 75 + 29 = 183
np.save('./Liz/features_liz_1.npy', flatten[0:79])
np.save('./Liz/features_liz_2.npy', flatten[79:79+75])
np.save('./Liz/features_liz_3.npy', flatten[79+75:79+75+29])
