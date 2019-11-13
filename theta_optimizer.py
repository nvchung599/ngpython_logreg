from general import *


class ThetaOptimizer(object):

    def __init__(self, it_max_input, dJ_stable_target, alpha_input):
        self.it_max = it_max_input
        self.dJ_stable = dJ_stable_target
        self.J_history = []
        self.alpha = alpha_input
        self.theta_opt = None

    def optimize_theta(self, X, y, reg_const):
        it = 0
        dJ_current = 999
        theta = construct_theta(X)

        while it < self.it_max and dJ_current > self.dJ_stable:
            self.J_history = np.append(self.J_history, calc_cost(X, y, theta, reg_const))
            grad = calc_grad(X, y, theta, reg_const)
            if it > 0:
                dJ_current = self.J_history[it - 1] - self.J_history[it]
            it = it + 1
            theta = theta - self.alpha * grad
        self.theta_opt = theta
        return (self.theta_opt, self.J_history[-1])

    def plot_last(self):
        print('theta_optimized:')
        print(self.theta_opt)
        print('J_optimized:')
        print(self.J_history[-1])
        plt.title('J History')
        plt.xlabel('iteration')
        plt.ylabel('cost')
        plt.plot(self.J_history)
        plt.show()

    def plot_dec_bound(self, X, theta, threshold):
        """X is the raw data set, sampled to find min/max values
        theta should be optimized already
        threshold is typically set at 0.5"""
        x1_min = np.min(X[:,0])
        x1_max = np.max(X[:,0])
        x2_min = np.min(X[:,1])
        x2_max = np.max(X[:,1])

        x1_sparse = np.arange(x1_min, x1_max, 1000)
        x2_sparse = np.arange(x2_min, x2_max, 1000)
        x1_sparse = x1_sparse.reshape(-1,1)
        x2_sparse = x2_sparse.reshape(-1,1)
        X_sparse = np.hstack([x1_sparse, x2_sparse])
        print(X)

    def check_my_gradients(self, X, y, reg_const, epsilon):
        theta = construct_theta(X)
        grad_check(X, y, theta, epsilon, reg_const)
