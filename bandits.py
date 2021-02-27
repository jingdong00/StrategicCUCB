from __future__ import division

import time
import numpy as np


class Bandit(object):

    def generate_reward(self, i):
        raise NotImplementedError


class BernoulliBandit(Bandit):
    def __init__(self, n, probas=None, budget = None, LSI = True):
        assert probas is None or len(probas) == n
        self.n = n
        self.B = [0 for i in range(n)]
        self.count = [0 for i in range(n)]
        self.LSI = LSI
        if probas is None:
            np.random.seed(int(time.time()))
            self.probas = [np.random.random() for _ in range(self.n)]
        else:
            self.probas = probas

        self.best_proba = max(self.probas)
        self.best_arms = (-np.array(self.probas)).argsort()[:2]
    
    def set_budget(self, budget):
        self.B = budget

    def generate_reward(self, i):
        # The player selected the i-th machine.
        if np.random.random() < self.probas[i]:
            reward = 1
        else:
            reward = 0
        
        if self.LSI:
            if self.count[i] == 0:
                self.count[i] += 1
                return reward + self.B[i]
            else:
                self.count[i] += 1
                return reward
        else: #Non-LSI strategy
            if self.B[i] > 0:
                strategy = np.random.random()
                if self.B[i] - strategy < 0:
                    return reward
                else:
                    self.B[i] -= strategy
                    return reward + strategy
            else:
                return reward

        
        
            
    
    def combinatorial_best(self, card = 2):
        best_arms = (-np.array(self.probas)).argsort()[:card]
        print("best arms", best_arms)
        total_prob = 0
        for a in best_arms:
            total_prob += self.probas[a]
        print(self.probas)
        return total_prob

#To evaluate collusion strategy
class other_Bandit(Bandit):

    def __init__(self, n, probas=None, budget = None, ptype = ''):
        assert probas is None or len(probas) == n
        self.n = n
        self.B = [0 for i in range(n)]
        self.count = [0 for i in range(n)]
        if probas is None:
            np.random.seed(int(time.time()))
            self.probas = [np.random.random() for _ in range(self.n)]
        else:
            self.probas = probas

        self.best_proba = max(self.probas)
        self.best_arms = (-np.array(self.probas)).argsort()[:2]
        self.ptype = ptype
    
    def set_budget(self, budget):
        self.B = budget
        if self.ptype == 'B':
            self.priority = (np.argsort(self.B)[::-1]).tolist()
        elif self.ptype == 'delta':
            self.priority = np.argsort(self.probas).tolist()
        elif self.ptype == 'Bdelta':
            self.priority = (np.argsort(np.add(self.B, self.probas))[::-1]).tolist()

    def generate_reward(self, i):
        # The player selected the i-th machine.
        if np.random.random() < self.probas[i]:
            reward = 1
        else:
            reward = 0
        
        #LSI strategy
        if self.count[i] == self.priority.index(i) + 1:
            self.count[i] += 1
            return reward + self.B[i]
        elif self.count[i] == 1 and self.B[i] > self.best_proba - self.probas[i]:
            self.count[i] += 1
            self.B[i] -= self.best_proba - self.probas[i]
            return self.best_proba
        else:
            self.count[i] += 1
            return reward
        #        return reward + strategy
            
    
    def combinatorial_best(self, card = 5):
        best_arms = (-np.array(self.probas)).argsort()[:2]
        print("best arms", best_arms)
        total_prob = 0
        for a in best_arms:
            total_prob += self.probas[a]
        print(self.probas)
        return total_prob


