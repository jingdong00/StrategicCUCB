import matplotlib  # noqa
matplotlib.use('Agg')  # noqa

import matplotlib.pyplot as plt
import numpy as np

from bandits import BernoulliBandit, other_Bandit
from solvers import Solver, CUCB, naive_CUCB


def plot_results(solvers, solver_names, figname):
    """
    Plot the results by multi-armed bandit solvers.

    Args:
        solvers (list<Solver>): All of them should have been fitted.
        solver_names (list<str)
        figname (str)
    """
    assert len(solvers) == len(solver_names)
    assert all(map(lambda s: isinstance(s, Solver), solvers))
    assert all(map(lambda s: len(s.regrets) > 0, solvers))

    b = solvers[0].bandit

    fig = plt.figure(figsize=(8, 8))
    fig.subplots_adjust(bottom=0.15, wspace=0.3)

    ax1 = fig.add_subplot(111)

    # Sub.fig. 1: Regrets in time.
    linestyle = ['-', ':', '--', 'dashed', 'solid', '-.', '-', ':', '--', 'dashed']
    markers = ['^','o','+','*','s', '^','o','+','*','s']
    for i, s in enumerate(solvers):
        print(s.regrets[-1])
        ax1.plot(range(len(s.pulls)), s.pulls, label=solver_names[i], linestyle = linestyle[i], linewidth=2)
        #ax1.plot(range(len(s.regrets)), s.regrets, label=solver_names[i], linestyle = linestyle[i], linewidth=2)
   
    ax1.set_xlabel('Time horizon', fontsize = 18)
    ax1.set_ylabel('Suboptimal pulls', fontsize = 18)
    #ax1.set_ylabel('Cumulative regret', fontsize = 18)
    plt.tick_params(axis='both', which='minor', labelsize=16)
    plt.title("Effect of Collusion Strategies on \n Number of Suboptimal Pulls",  fontsize = 20)
    #plt.title("Effect of Collusion Strategies on Cumulative Regret", fontsize = 20)
    ax1.legend(fontsize  = 15)
    #ax1.legend(loc=9, bbox_to_anchor=(1.82, -0.25), ncol=5)
    ax1.grid('k', ls='--', alpha=0.3)
    
    params = {
        'axes.labelsize': 8,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'text.usetex': False,
        'figure.figsize': [4.5, 4.5]
    }
    #plt.rcParams.update(params)

    plt.savefig(figname)


def experiment(K, N):
    """
    Run a small experiment on solving a Bernoulli bandit with K slot machines,
    each with a randomly initialized reward probability.

    Args:
        K (int): number of slot machiens.
        N (int): number of time steps to try.
    """
    #prob = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.95]
    b = BernoulliBandit(K, 
    probas=[0.4712092996727222, 0.7545497980220683, 0.33869624139568244, 0.5975096556916595, 0.9424158319815205, 0.1441794410691316, 0.4858694814807426, 0.9349580583309945, 0.03602531689225663, 0.2411479721284544])
    print("Randomly generated Bernoulli bandit has reward probabilities:\n"), b.probas

   

    #randomly assign budget, maximum budget = 10
    budget = np.random.randint(50, size=K) #
    #best_arms = (-np.array(b.probas)).argsort()[:2]
    Bmax = np.max(budget)
    b.set_budget(budget)

    b2 = other_Bandit(K, probas=b.probas, ptype = 'B')
    b2.set_budget(budget)
    b3 = other_Bandit(K, probas=b.probas, ptype = 'delta')
    b3.set_budget(budget)
    b4 = other_Bandit(K, probas=b.probas, ptype = 'Bdelta')
    b4.set_budget(budget)
    #b.set_budget(budget)

    test_solvers = [
        naive_CUCB(b, Bmax=Bmax),
        naive_CUCB(b2, Bmax=Bmax),
        naive_CUCB(b3, Bmax=Bmax),
        naive_CUCB(b4, Bmax=Bmax),
    ]
    names = [
        'LSI',
        'PB-LSI',
        'PD-LSI',
        'PBD-LSI'  
    ]

    for s in test_solvers:
        regret = [0 for i in range(N)]
        #average over 10 trials
        for i in range(10):
            s.clear()
            s.run(N)
            regret = np.add(regret, s.pulls[:N])
            #print(s.regrets[:N], regret)
        s.pulls = np.divide(regret, 10)

    plot_results(test_solvers, names, "results_K{}_N{}.png".format(K, N))


if __name__ == '__main__':
    experiment(10, 5000)
