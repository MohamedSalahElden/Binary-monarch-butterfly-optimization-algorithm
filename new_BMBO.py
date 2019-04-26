import numpy as np
from data_set import load_data_set
from utility_functions import arrange_by_density
import matplotlib.pyplot as plt
from scipy.stats import levy
import math
import random



Maxgen      = 50
Smax        = 1
t           = 1
P           = 5/12
p           = 0.5
BAR         = 5/12
NP          = 10
NP1         = 4
NP2         = 6
peri        = 1.2
c           = 1000
no_of_items = 100

def calculate_value_and_weight(x , v , w):
	d = int((x.shape[1]-2) /2)
	NP = x.shape[0]
	for i in range(NP):
		total_value = 0
		total_weight = 0
		for j in range(d,d*2):
			total_value 	= total_value 		+ x[i,j] * v[j-d]
			total_weight = total_weight  	+ x[i,j] * w[j-d]
		weight_index = d*2 + 1
		value_index = d*2 
		x[i,value_index] 	= int(total_value)
		x[i,weight_index] 	= int(total_weight)
	return x 
def sigmoid(x):
	y = 1/(1+np.exp(-x))
	return y
def Real2Binary(x):
	d = int((x.shape[1]-2) /2)
	for i in range(x.shape[0]):
		for j in range(d,d*2):
			if sigmoid(x[i,j-d]) >= 0.5:
				x[i,j] = 1
			else :
				x[i,j] = 0
	return x
def generate_random_population(NP , V, W ):						#return population , values , weights
	d = V.shape[0] 
	data = np.empty([NP , d*2+2])
	for i in range(NP):
		for j in range(d):
			data[i,j] = random.uniform(-3,3)
		# population[i , :n] = np.random.choice([0, 1], size=(n,), p=[3./4, 1./4] )
	Real2Binary(data)
	calculate_value_and_weight(data, v, w)
	return data
def sort_by_fitness(x):
	x = x[x[:,2*4].argsort()]
	x = np.flip(x,0)
	return x
def migration_operator(subpop1 , subpop2 , peri , p , v , w):				#return subpop1
	NP1 = subpop1.shape[0]			#number of monarches in subpopulation1
	d   = int((subpop1.shape[1]-2)/2)			#number of elements in monarch individual
	NP2 = subpop2.shape[0]
	p = NP1 / (NP1 + NP2)
	r = 0
	r1 = 0
	r2 = 0
	for i in range(NP1):		#for all monarch in subpopulation 1
		for j in range(d):		#for all elements in monrach number of elements = D
			r = np.random.uniform(0 , 1) * peri
			if r <= p:
				r1 = np.random.randint(0,NP1)
				subpop1[i,j] =  subpop1[r1,j]
			else:
				r2 = np.random.randint(0,NP2)
				subpop1[i,j] =  subpop2[r2,j]
	subpop1 = update_array(subpop1 , v, w)
	return subpop1

def update_array(x , v ,w):
	x = Real2Binary(x)
	x = calculate_value_and_weight(x, v, w)
	return x
def butterfly_adjusting_operator(subpopulation2 , Xbest , Maxgen , Smax , t , p , BAR , v , w) :		#return subpopulation2 
	NP2 = subpopulation2.shape[0]
	d   = int((subpopulation2.shape[1] - 2) / 2)
	dx = np.empty(d)
	omega = Smax / (t**2)		#t is the current generation 
	for i in range(d):
		StepSize = math.ceil(np.random.exponential(2 * Maxgen)) 
		dx[i]    = levy.pdf( StepSize )
	for i in range(NP2):		#for all monarch in subpopulation 2
		for j in range(d):		# for all elements in monarch 
			rand = np.random.uniform(0 , 1)
			if rand <= p:		
				subpopulation2[i,j] = Xbest[j]
			else:
				r3 = np.random.randint(0,NP2)
				subpopulation2[i,j] = subpopulation2[r3,j]
				if rand > BAR:
					subpopulation2[i,j] = subpopulation2[i,j] + omega * (dx[j] - 0.5) 
	
	update_array(subpopulation2, v, w)
	return subpopulation2 
def Greedy_Optimization_Algorithm(X , W , V , H , C):     			#return X , value , weight
	#step 1 : repair
	
	n = H.shape[0]	 # H array is the arranged items indices by their capacity 
	d = int((X.shape[0] -2)/2)
	weight = 0 
	value = 0
	temp = 0

	#calculate the waight 
	for i in range(d):
		weight = weight + X[i+d] * W[i]
	

	if (weight>C):										#check if wight exceed the knap sack capacity 
		for i in range(d):								#for every item in the search area 
			temp = temp + X[H[i]+d] * W[H[i]]				#take the next highest dencity item in H array 
														#add it to the temp weight variable
			if (temp > C):								#if the selected item exceed that weight 
				temp = temp - X[H[i]+d] * W[H[i]]			#leave that weight again 
				X[H[i]+d] = 0								#put 0 in the binary X matrix at its the position
				X[H[i]] = -3		#update the real values 
		weight = temp 									#current weight of the knapsack 
	
	#step 2 : optimize

	for i in range(d):									#for all elements in X[H]
		if (X[H[i]+d] == 0) and ((weight + W[H[i]]) <= C):	#
			X[H[i]+d] = 1 		#update the binary values 
			X[H[i]] = 0			#update the real values 
			weight = weight + X[H[i]+d] * W[H[i]]	
	#step 3 : compute

	for i in range(d):
		value = value + X[i+d] * V[i]
	

	# result = np.append(X, [value , weight])
		
	return X	


v , w   = load_data_set(no_of_items)
H = arrange_by_density(w, v)
gen = generate_random_population(100 , v, w )
# print(gen[:,4:])
# for i in range(gen.shape[0]):
# 	gen[i] = Greedy_Optimization_Algorithm(gen[i] , w , v , H , c)
# print("---------------------------------------------------------")
# update_array(gen , v ,w)

# print(gen[:,4:])
# gen = sort_by_fitness(gen)
# subpop_1 = gen[:3,:]
# subpop_2 = gen[3:6,:]
# print(subpop_1[:,4:])
# print("#############################################")
# print(subpop_2[:,4:])
# print("#############################################")
# subpop_1 = migration_operator(subpop_1, subpop_2, peri, p , v , w)
# subpop_2 = butterfly_adjusting_operator(subpop_2, gen[0,:], Maxgen, Smax, t, p, BAR , v , w)
# print(subpop_1[:,4:])
# # print("#############################################")
# print(subpop_2[:,4:])
# d=4
# sub_pop_bin = subpop_1[:,d:2*d]

# 1 load data set and construct population
v , w   = load_data_set(no_of_items)
H = arrange_by_density(w, v)
gen = generate_random_population(10 , v, w )
# print(gen)

sol = np.empty(Maxgen)

# 2 start algorithm
while t <= Maxgen:
	gen = sort_by_fitness(gen)

	subpop_1 = gen[:NP1,:]
	subpop_2 = gen[NP1:,:]

	best = gen[0,:]

	subpop_1 = migration_operator(subpop_1, subpop_2, peri, p , v , w)
	subpop_2 = butterfly_adjusting_operator(subpop_2, best, Maxgen, Smax, t, p, BAR , v , w)
	gen = np.concatenate((subpop_1, subpop_2))
	for i in range(gen.shape[0]):
		gen[i] = Greedy_Optimization_Algorithm(gen[i] , w , v , H , c)
	update_array(gen , v ,w)
	gen = sort_by_fitness(gen)
	best_solution = gen[0]
	sol[t-1] = best_solution[no_of_items*2]
	print(best_solution[no_of_items*2:])
	t = t+1
	pass
print(sol)
plt.plot(sol)
plt.show()