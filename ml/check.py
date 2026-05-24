import numpy as np 
names = np.load('../backend/data/scalograms/names.npy') 
targets = ['v100s', 'v101l', 'a103l', 'a104s', 't107l', 'b124s'] 
[print(t, 'FOUND' if t in names else 'NOT FOUND') for t in targets] 
