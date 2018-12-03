import numpy as np
import arms
import matplotlib.pyplot as plt
from utils import UCB1,TS

# Build your own bandit problem

# this is an example, please change the parameters or arms!
arm1 = arms.ArmBernoulli(0.70, random_state=np.random.randint(1, 312414))
arm2 = arms.ArmBernoulli(0.30, random_state=np.random.randint(1, 312414))
arm3 = arms.ArmBernoulli(0.10, random_state=np.random.randint(1, 312414))
arm4 = arms.ArmBernoulli(0.40, random_state=np.random.randint(1, 312414))


# arm1 = arms.ArmExp(5, random_state=np.random.randint(1, 312414))
# arm2 = arms.ArmExp(0.7, random_state=np.random.randint(1, 312414))
# arm3 = arms.ArmExp(0.01, random_state=np.random.randint(1, 312414))
# arm4 = arms.ArmExp(1, random_state=np.random.randint(1, 312414))

MAB = [arm1, arm2, arm3, arm4]

# bandit : set of arms

nb_arms = len(MAB)
means = [el.mean for el in MAB]

# Display the means of your bandit (to find the best)
print('means: {}'.format(means))
mu_max = np.max(means)

# Comparison of the regret on one run of the bandit algorithm
# try to run this multiple times, you should observe different results

T = 5000  # horizon
nb_simu = 50

rew1, draws1 = UCB1(T, MAB,nb_simu) # rew is reward
reg1 = mu_max * np.arange(1, T + 1) - np.mean(np.cumsum(rew1,axis = 1),axis = 0)
rew2, draws2 = TS(T, MAB, nb_simu)
reg2 = mu_max * np.arange(1, T + 1) - np.mean(np.cumsum(rew2,axis = 1),axis = 0)


best_arm = np.argmax(means)
MAB_filtered = list(set(MAB).difference(set([MAB[best_arm]])))

def KL(x,y):
	output = x*np.log(x/y) + (1-x)*np.log((1-x)/(1-y))
	return output

C_p = np.sum([(mu_max - el.mean)/KL(mu_max,el.mean) for el in MAB_filtered])
oracle_regret = [C_p*np.log(t) for t in range(1,T+1)]


plt.figure(1)
x = np.arange(1, T+1)
plt.plot(x, reg1, label='UCB')
plt.plot(x, reg2, label='Thompson')
plt.plot(x,oracle_regret, label = 'Oracle')
plt.xlabel('Rounds')
plt.ylabel('Cumulative Regret')
plt.legend()
plt.show()
