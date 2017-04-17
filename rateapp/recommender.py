from numpy import *



num_movies = 10
num_users = 5

MOVIE_RATINGS=array([[1,7,1,9,1]
,[0,8,5,0,8]
,[7,0,0,3,5]
,[3,2,1,0,0]
,[0,7,8,1,0]
,[2,4,1,6,7]
,[4,0,9,9,0]
,[4,1,0,0,8]
,[0,9,0,6,0]
,[7,5,9,6,10]])

MOVIE_RATINGS2=random.randint(5,11, size = (num_movies, num_users))

def normalize_ratings(ratings, did_rate):
    num_movies = ratings.shape[0]
    
    ratings_mean = zeros(shape = (num_movies, 1))
    ratings_norm = zeros(shape = ratings.shape)
    
    for i in range(num_movies): 
        # Get all the indexes where there is a 1
        idx = where(did_rate[i] == 1)[0]
        #  Calculate mean rating of ith movie only from user's that gave a rating
        ratings_mean[i] = mean(ratings[i, idx])
        ratings_norm[i, idx] = ratings[i, idx] - ratings_mean[i]
    
    return ratings_norm, ratings_mean



def unroll_params(X_and_theta, num_users, num_movies, num_features):
	# Retrieve the X and theta matrixes from X_and_theta, based on their dimensions (num_features, num_movies, num_movies)
	# --------------------------------------------------------------------------------------------------------------
	# Get the first 30 (10 * 3) rows in the 48 X 1 column vector
	first_30 = X_and_theta[:num_movies * num_features]
	# Reshape this column vector into a 10 X 3 matrix
	X = first_30.reshape((num_features, num_movies)).transpose()
	# Get the rest of the 18 the numbers, after the first 30
	last_18 = X_and_theta[num_movies * num_features:]
	# Reshape this column vector into a 6 X 3 matrix
	theta = last_18.reshape(num_features, num_users ).transpose()
	return X, theta


# In[59]:

def calculate_gradient(X_and_theta, ratings, did_rate, num_users, num_movies, num_features, reg_param):
	X, theta = unroll_params(X_and_theta, num_users, num_movies, num_features)
	
	# we multiply by did_rate because we only want to consider observations for which a rating was given
	difference = X.dot( theta.T ) * did_rate - ratings
	X_grad = difference.dot( theta ) + reg_param * X
	theta_grad = difference.T.dot( X ) + reg_param * theta
	
	# wrap the gradients back into a column vector 
	return r_[X_grad.T.flatten(), theta_grad.T.flatten()]


# In[60]:

def calculate_cost(X_and_theta, ratings, did_rate, num_users, num_movies, num_features, reg_param):
	X, theta = unroll_params(X_and_theta, num_users, num_movies, num_features)
	
	# we multiply (element-wise) by did_rate because we only want to consider observations for which a rating was given
	cost = sum( (X.dot( theta.T ) * did_rate - ratings) ** 2 ) / 2
	# '**' means an element-wise power
	regularization = (reg_param / 2) * (sum( theta**2 ) + sum(X**2))
	return cost + regularization





def do_stuff(my_ratings):
	global MOVIE_RATINGS
	ratings = MOVIE_RATINGS
	print ratings,"hi"

	did_rate = (ratings != 0) * 1
	print did_rate

	# print (ratings != 0)

	# print (ratings != 0) * 1


	# my_ratings = zeros((num_movies, 1))
	# # print my_ratings
	# my_ratings[0] = 8
	# my_ratings[4] = 7
	# my_ratings[7] = 3

	# print my_ratings

	ratings = append(my_ratings, ratings, axis = 1)
	did_rate = append(((my_ratings != 0) * 1), did_rate, axis = 1)


	a = [10, 20, 30]
	aSum = sum(a)

	aMean = aSum / 3

	aMean = mean(a)
	# print aMean

	a = [10 - aMean, 20 - aMean, 30 - aMean]

	# a function that normalizes a dataset



	ratings, ratings_mean = normalize_ratings(ratings, did_rate)

	num_users = ratings.shape[1]
	num_features = 3

	X = array([[1, 2], [1, 5], [1, 9]])
	Theta = array([[0.23], [0.34]])

	# print X

	# print Theta

	Y = X.dot(Theta)
	# print Y

	movie_features = random.randn( num_movies, num_features )
	user_prefs = random.randn( num_users, num_features )
	initial_X_and_theta = r_[movie_features.T.flatten(), user_prefs.T.flatten()]

	# print movie_features

	# print initial_X_and_theta


	initial_X_and_theta.shape

	movie_features.T.flatten().shape

	user_prefs.T.flatten().shape


	initial_X_and_theta

	# In[64]:

	# import these for advanced optimizations (like gradient descent)

	from scipy import optimize

	reg_param = 30

	# perform gradient descent, find the minimum cost (sum of squared errors) and optimal values of X (movie_features) and Theta (user_prefs)

	minimized_cost_and_optimal_params = optimize.fmin_cg(calculate_cost, fprime=calculate_gradient, x0=initial_X_and_theta, 								args=(ratings, did_rate, num_users, num_movies, num_features, reg_param), 								maxiter=100, disp=True, full_output=True ) 

	cost, optimal_movie_features_and_user_prefs = minimized_cost_and_optimal_params[1], minimized_cost_and_optimal_params[0]

	# unroll once again

	movie_features, user_prefs = unroll_params(optimal_movie_features_and_user_prefs, num_users, num_movies, num_features)

	# print movie_features


	# print user_prefs

	# Make some predictions (movie recommendations). Dot product

	all_predictions = movie_features.dot( user_prefs.T )


	# print all_predictions

	predictions_for_my = all_predictions[:, 0:1] + ratings_mean

	dic = {}

	for x in range(10):
		if my_ratings[x] != 0:
			continue
		dic['movie'+str(x)] = predictions_for_my[x]
	# print dic
	sorted_dic = sorted(dic, key=dic.get, reverse=True)
	return sorted_dic
