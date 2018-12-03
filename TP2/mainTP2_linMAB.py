import numpy as np
from linearmab_models import ToyLinearModel, ColdStartMovieLensModel
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

random_state = np.random.randint(0, 24532523)
# model = ToyLinearModel(
#     n_features=8,
#     n_actions=20,
#     random_state=random_state,
#     noise=0.1)

model = ColdStartMovieLensModel(
    random_state=random_state,
    noise=0.1
)

n_a = model.n_actions
print('number of actions ',n_a)
d = model.n_features

T = 6000

nb_simu = 5 # you may want to change this!


def LinearUCB(features,previous_rewards,previous_actions,eps_greedy,epsilon = None,lambda_=0.1,d=d,n_a=n_a):
    """
    Parameters
    ------------
    features : features of the movies
    previous_rewards  : rewards received up to time t 
    previous_actions : actions taken up to time t
    eps_greedy : Use eps-greedy strategy if True
    epsilon : parameter of eps-greedy strategy
    lambda_ : regularization parameter for Ridge
    d : dimension of the features
    """

    Z = features[previous_actions]

    y = previous_rewards

    theta_hat = np.dot(np.linalg.inv((Z.T)@Z + lambda_*np.identity(d))@(Z.T),y)
 
    t = len(y) + 1

    alpha = np.sqrt(d*np.log((1+t*n_a/lambda_)))+np.sqrt(lambda_)*np.linalg.norm(model.real_theta,2)

    beta = [alpha*np.sqrt(np.dot(features[action_],np.dot(np.linalg.inv((Z.T)@Z + lambda_*np.identity(d)),features[action_]))) for action_ in range(len(features))]

    if eps_greedy:
        if random.uniform(0,1) > epsilon:
            action = np.argmax([np.dot(features[action_],theta_hat) + beta[action_] for action_ in range(len(features))])
        else:
            action = np.random.choice(list(range(len(features))))

    else:
        action = np.argmax([np.dot(features[action_],theta_hat) + beta[action_] for action_ in range(len(features))])

    return action,theta_hat


# alphas = [2]
# nb_alphas = len(alphas)

# regret = np.zeros((nb_alphas,nb_simu, T))
# norm_dist = np.zeros((nb_alphas,nb_simu, T))

regret = np.zeros((nb_simu, T))
norm_dist = np.zeros((nb_simu, T))

np.random.seed(42)

alg_name = 'LinearUCB'
# alg_name = 'Random'
# alg_name = 'Eps_Greedy'

# for i in range(len(alphas)):

for k in tqdm(range(nb_simu), desc="Simulating {}".format(alg_name)):
    actions = [np.random.choice(list(range(d)))]
    rewards = [model.reward(actions[0])]

    for t in range(T):
        if alg_name == 'LinearUCB':
            a_t,theta_hat = LinearUCB(model.features,rewards,actions,False)
            norm_dist[k, t] = np.linalg.norm(theta_hat - model.real_theta, 2)

        elif alg_name == 'Random':
            a_t = np.random.choice(list(range(n_a)))

        elif alg_name == 'Eps_Greedy':
            a_t,theta_hat = LinearUCB(model.features,rewards,actions,True,0.1)
            norm_dist[k, t] = np.linalg.norm(theta_hat - model.real_theta, 2)

        r_t = model.reward(a_t) # get the reward

        # update algorithm
        actions.append(a_t)
        rewards.append(r_t)

        # store regret
        regret[k, t] = model.best_arm_reward() - r_t



# compute average (over sim) of the algorithm performance and plot it
# for i in range(len(alphas)):

# mean_regret = np.mean(regret[i],axis = 0)

mean_regret = np.mean(regret,axis = 0)

if alg_name == 'LinearUCB' or alg_name == 'Eps_Greedy':

    mean_norms = np.mean(norm_dist,axis = 0)

    plt.figure(1)
    plt.subplot(121)
    plt.plot(mean_norms, label=alg_name)
    # plt.plot(mean_norms, label=alg_name+' '+str(alphas[i]))
    plt.ylabel('d(theta, theta_hat)')
    plt.xlabel('Rounds')
    plt.legend()

    plt.subplot(122)
    plt.plot(mean_regret.cumsum(), label=alg_name)
    # plt.plot(mean_regret.cumsum(), label=alg_name+' '+str(alphas[i]))
    plt.ylabel('Cumulative Regret')
    plt.xlabel('Rounds')
    plt.legend()
    plt.show()
    # plt.savefig('C:/Users/Idriss/Desktop/3A/MVA/RL/TPs/TP2/TP2_python/'+alg_name+' graph for alpha ='+str(alphas[i])+'.jpg')

else:
    plt.plot(mean_regret.cumsum())
    # plt.plot(mean_regret.cumsum(), label=alg_name+' '+str(alphas[i]))
    plt.ylabel('Cumulative Regret')
    plt.xlabel('Rounds')
    plt.legend()
    plt.show()
    # plt.savefig('C:/Users/Idriss/Desktop/3A/MVA/RL/TPs/TP2/TP2_python/'+alg_name+' graph for alpha ='+str(alphas[i])+'.jpg')

