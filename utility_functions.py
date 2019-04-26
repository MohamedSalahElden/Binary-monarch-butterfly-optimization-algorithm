import numpy as np

def arrange_by_density(W , V):
	dim = W.shape[0]
	index = np.array(range(W.shape[0])).reshape(W.shape[0] , 1)
	dinsity = V / W
	H = np.concatenate((dinsity, index), axis = 1)
	H = H[np.argsort(H[:,0])]
	H = np.flip(H,0)
	H = H[:,1]
	result = list(map(int, H))
	result = np.array(result)
	result = result.reshape(dim,1)
	return result



