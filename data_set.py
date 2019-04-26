import numpy as np



def load_data_set(num):
	allNums = []
	#D:\desktop 6\SALAH\MSc\algorithms-2\proj\instances_01_KP\low-dimensional
	#file_name = "instances_01_KP\large_scale\knapPI_1_" + str(num) + "_1000_1"
	file_name = "instances_01_KP\\low-dimensional\\f4_l-d_kp_4_11"
	# num_of_elements =int ( file_name.split("_")[5])
	f = open(file_name, "r+")

	data = f.readlines() # read the text file
	    
	for line in data:
	    allNums += line.strip('\n').split(" ") # get a list containing all the numbers in the file
	#print(allNums)
	allNums = list(map(int, allNums))
	data = np.array(allNums[2:])
	
	V_W = data[:num * 2].reshape((num , 2))
	#print(V_W.shape)
	#X   = data[ num*2 : ].reshape((num , 1))
	
	#print(X.shape)
	# dataset = np.concatenate((V_W , X), axis = 1)
	#index = index.reshape((num_of_elements,1))
	values = np.array(V_W[: , 0])
	values = values.reshape(num , 1)
	weights = np.array(V_W[: , 1])
	weights = weights.reshape(num , 1)
	# x = np.array(X)
	return values , weights		# , X

#print( load_data_set(4))
	
	




















#f = open("instances_01_KP\large_scale\knapPI_1_100_1000_1", "r")
#x = f.read()
#print(x)
#print(x[0] ,x[1],x[2] )
