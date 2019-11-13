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
        dJ_current = 99999999999999999999999
        theta = construct_theta(X)

        #while it < self.it_max and dJ_current > self.dJ_stable:
        while it < self.it_max:
            self.J_history = np.append(self.J_history, calc_cost(X, y, theta, reg_const))
            grad = calc_grad(X, y, theta, reg_const)
            if it > 0:
                dJ_current = self.J_history[it - 1] - self.J_history[it]
                print(dJ_current)
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

    def plot_dec_bound(self, X, y,theta, threshold=0.5, margin=0.001):
        """X is the raw data set, sampled to find min/max values
        theta should be optimized already
        threshold is typically set at 0.5
        margin is the decision line width about the threshold, use 0.05"""
        x1_min = np.min(X[:,0])
        x1_max = np.max(X[:,0])
        x2_min = np.min(X[:,1])
        x2_max = np.max(X[:,1])

        x1_sparse = np.arange(x1_min, x1_max, 0.1)
        x2_sparse = np.arange(x2_min, x2_max, 0.1)
        xx1, xx2 = np.meshgrid(x1_sparse, x2_sparse)
        X_grid_raw = np.dstack([xx1, xx2]).reshape(-1, 2)

        X_grid = normalize(X_grid_raw)
        X_grid = add_bias(X_grid)

        hypo_grid = calc_hypo(X_grid, theta)
        mask_part1 = hypo_grid>(threshold-margin)
        mask_part2 = hypo_grid<(threshold+margin)
        mask = mask_part1 == mask_part2
        mask = mask.reshape(-1)
        dec_grid = X_grid_raw[mask]

        plt.plot(dec_grid[:,0], dec_grid[:,1], marker='x')

        pos_i = y == 1
        pos_i = pos_i.reshape(-1)
        neg_i = y == 0
        neg_i = neg_i.reshape(-1)
        X_pos = X[pos_i, :]
        X_neg = X[neg_i, :]

        plt.title('Data Visualization')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.scatter(X_pos[:, 0], X_pos[:, 1], marker='+', label='positive')
        plt.scatter(X_neg[:, 0], X_neg[:, 1], marker='o', label='negative')

        plt.show()

    def check_my_gradients(self, X, y, reg_const, epsilon):
        theta = construct_theta(X)
        grad_check(X, y, theta, epsilon, reg_const)
