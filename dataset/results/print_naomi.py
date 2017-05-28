import numpy as np

array = np.load('./Naomi/bottleneck_features_train.npy')
print ('Naomi')
print (np.shape(array))

flatten = array.reshape((array.shape[0], 4*4*512))

# 63 + 42 + 31 = 136
np.save('./Naomi/features_naomi_1.npy', flatten[0:63])
np.save('./Naomi/features_naomi_2.npy', flatten[63:63+42])
np.save('./Naomi/features_naomi_3.npy', flatten[63+42:63+42+31])
