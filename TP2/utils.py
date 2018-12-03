import numpy as np
import arms

def UCB1(T,MAB):
	"""
	T : Horizon
	MAB : list of arms
	n_simu : number of simulations
	"""

	rews = np.zeros(T)
	draws = np.zeros(T)


	K = len(MAB)

	nb_pulls = [1]*K
	cumreward_arm = []

	for k in range(K):
		reward = int(MAB[k].sample())
		cumreward_arm.append(reward)

	for t in range(1,T+1): 

		scores = [cumreward_arm[i]/nb_pulls[i] + 0.2*np.sqrt(np.log(t)/(2*nb_pulls[i])) for i in range(len(MAB))]

		best_arm = np.argmax(scores)

		sample = MAB[best_arm].sample()

		if isinstance(sample,bool):
			best_reward = int(sample)
		else:
			best_reward = sample[0]

		rews[t-1] = best_reward
		draws[t-1] = best_arm

		#Update
		nb_pulls[best_arm] += 1
		cumreward_arm[best_arm] += best_reward

	return rews,draws



def TS(T,MAB):
	"""
	T : Horizon
	MAB : list of arms
	n_simu : number of simulations
	"""

	rews = np.zeros(T)
	draws = np.zeros(T)

	K = len(MAB)

	nb_pulls = [1]*len(MAB)
	cumreward_arm = []

	for k in range(K):
		reward = int(MAB[k].sample())
		cumreward_arm.append(reward)

	for t in range(1,T+1): 

		mu = [np.random.beta(cumreward_arm[i]+1,nb_pulls[i]-cumreward_arm[i]+1) for i in range(len(MAB))]

		best_arm = np.argmax(mu)

		sample = MAB[best_arm].sample()

		if isinstance(sample,bool):
			reward = int(sample)
			success = reward
		else:
			reward = sample[0]
			arms_object = arms.ArmBernoulli(p = reward)
			success = int(arms_object.sample())

		#Update
		nb_pulls[best_arm] += 1
		cumreward_arm[best_arm] += success

		rews[t-1] = reward
		draws[t-1] = best_arm

	return rews,draws



