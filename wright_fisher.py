import numpy as np
import pylab as pp
import time

from problems import ABC_Problem
import distributions as distr


def update_p0(k, x2, N):
    return 1.0 / (1.0 + k * x2 / N)


def d_x0(N, q):
    # print "N = %0.2f\tq = %f"%(N,q)
    if N == 0:
        return 0
    else:
        if q > 1e-6:
            if q < 1.0 and N > 0:
                return np.random.binomial(N, q)
            else:
                return N
        else:
            return 0


def d_x1(N, x0p, q_hat):
    if N - x0p == 0:
        return 0
    else:
        if q_hat > 1e-6:
            # print  N-x0p, q_hat
            if q_hat < 1.0 and int(N - x0p) > 0:
                return np.random.binomial(N - x0p, q_hat)
            else:
                return N - x0p
        else:
            return 0


def d_x2(N, x0p, x1p):
    return N - x0p - x1p


def simulate_all(T, N, k, p1, x0, x1, x2):
    T = int(T)
    CELLS = np.zeros((T, 3))

    # at time 0, set CELLS to initial values
    CELLS[0, :] = np.array([x0, x1, x2])
    kdivN = k / float(N)
    for t in range(T - 1):
        #p0 = update_p0( k, x2, N )
        p0 = 1.0 / (1.0 + kdivN * x2)
        if x0 + x1 > 0:
            q = x0 * p0 / (x0 + x1)
            # print "q = ", q
            # print x0+x1, q, 1.0-q

            if q < 1.0:
                q_hat = (x0 * (1 - p0) + x1 * p1) / ((x0 + x1) * (1 - q))

                x0p = d_x0(N, q)
                x1p = d_x1(N, x0p, q_hat)
            else:
                x0p = N
                x1p = 0
            x2p = d_x2(N, x0p, x1p)
        else:
            x0p = x0
            x1p = x1
            x2p = d_x2(N, x0p, x1p)

        x0 = x0p
        x1 = x1p
        x2 = x2p

        # print p0,q,q_hat,x0,x1,x2
        CELLS[t + 1, :] = np.array([x0, x1, x2])
    return CELLS


def cell_stats(Y):
    mu = Y.mean(0)
    var = Y.var(0)
    #mx = Y.max(0)
    # pdb.set_trace()
    return np.hstack((mu, np.log(var + 1e-12)))


def simulation(M, T, N, k, p1, x0, x1, x2):
    # params={'p1': 0.59, 'T': 500, 'x2': 1000,
    #'k': 8.2, 'x1': 1000, 'M': 50, 'x0': 1000, 'N': 981}
    #M = params["M"]
    #T = params["T"]
    #N = params["N"]
    #k = params["k"]
    #p1 = params["p1"]
    #x0 = params["x0"]
    #x1 = params["x1"]
    #x2 = params["x2"]

    Y = np.zeros((M, 3))
    oldtime = time.time()
    # print "sim: ",params
    for m in range(M):
        cell_ts = simulate_all(T, N, k, p1, x0, x1, x2)
        #Y[m,:] = np.hstack( (cell_ts[-1],cell_ts[5]))
        Y[m, :] = cell_ts[-1]  # + 1e-2*np.random.randn()
    finish_time = time.time()
    how_long = finish_time - oldtime
    oldtime = finish_time
    # print how_long
    return Y


def simulation_dict(params):
    # params={'p1': 0.59, 'T': 500, 'x2': 1000,
    #'k': 8.2, 'x1': 1000, 'M': 50, 'x0': 1000, 'N': 981}
    M = params["M"]
    T = params["T"]
    N = params["N"]
    k = params["k"]
    p1 = params["p1"]
    x0 = params["x0"]
    x1 = params["x1"]
    x2 = params["x2"]

    Y = np.zeros((M, 3))
    oldtime = time.time()
    # print "sim: ",params
    for m in range(M):
        cell_ts = simulate_all(T, N, k, p1, x0, x1, x2)
        #Y[m,:] = np.hstack( (cell_ts[-1],cell_ts[5]))
        Y[m, :] = cell_ts[-1]  # + 1e-2*np.random.randn()
    finish_time = time.time()
    how_long = finish_time - oldtime
    oldtime = finish_time
    print how_long
    return Y


class WF_proposal(object):

    def __init__(self, step, priors, priors_args):
        self.priors = priors
        self.priors_args = priors_args
        self.step = step

    def rvs(self, thetas):
        new_thetas = thetas.copy()
        D = len(thetas)
        I = np.random.permutation(D)

        if D == 1:
            I = [I[0]]
        elif D == 2:
            I = [I[0]]
        else:
            I = I[:2]

        #print "rate = ",rate
        #for t,theta, prior_cdf, prior_icdf in zip(range(D),thetas, prior_cdfs, prior_icdfs):
        for t in I:
            theta = thetas[t]
            prior = self.priors[t]
            prior_args = self.priors_args[t]

            u_theta = prior.cdf(theta, *prior_args)
            if np.random.randn() < 0:
                u_theta += self.step * np.random.rand()
            else:
                u_theta -= self.step * np.random.rand()

            if u_theta < 0:
                u_theta = -u_theta
                u_theta = u_theta % 1
            elif u_theta > 1:
                u_theta = 1.0 - u_theta % 1.0 #%1

            new_thetas[t] =  prior.icdf(u_theta, *prior_args )
        #print '\nproposed:', new_thetas
        return new_thetas

    def logpdf(self, theta1, theta2):
        return 0

    def pdf(self, theta1, theta2):
        return 1


class Wright_Fisher_Problem(ABC_Problem):

    def __init__(self, fixed=None):
        self.y_dim = 6
        # self.y_star = np.array(
        #    [741.818, 438.546,
        #     597.774, 9.21997747,
        #     7.64374539, 8.64950045])
        # self.y_star = np.array([8.278, 1014.442,
        #                        675.16, 8.31875316,
        #                        7.8853197, 7.69737124])
        self.y_star = np.array([180, 495, 1322, 7.75, 7.57, 9])

        #self.true_args = [1781.0, 0.1, 1.52, 300, 40, 10]
        #self.true_args = [1700, 0.6, 3, 300, 140, 200]
        self.true_true_args = [2000, 3, 0.1]
        self.simulator_args = ['N', 'k', 'p1']
        self.use_log = False

        self.x0 = 1000
        self.x1 = 1000
        self.x2 = 1000

        self.M = 10
        self.T = 500

        #[10.205, 1782.825,
        # 198.47, 8.3685592,
        # 9.76456763, 5.81681082])

        N_args = [1.5, 0.001]
        k_args = [2.0, 0.2]
        p1_args = [2.0, 2.0]

        prior_distrs = []
        prior_distr_args = []
        proposal_args = []

        self.simulator_args = []
        self.true_args = []

        if fixed is None:
            self.fixed = []
        else:
            self.fixed = fixed

        if 'N' not in self.fixed:
            prior_distrs.append(distr.gamma)
            prior_distr_args.append(N_args)
            self.simulator_args.append('N')
            self.true_args.append(2000)

        if 'k' not in self.fixed:
            prior_distrs.append(distr.gamma)
            prior_distr_args.append(k_args)
            self.simulator_args.append('k')
            self.true_args.append(3)

        if 'p1' not in self.fixed:
            prior_distrs.append(distr.beta)
            prior_distr_args.append(p1_args)
            self.simulator_args.append('p1')
            self.true_args.append(0.1)

        self.prior = distr.multivariate_mixture(prior_distrs, prior_distr_args)

        self.prior_args = []
        #self.prior = distr.uniform_nd
        # self.prior_args = [np.array([1500, 0, 1]),
        #                   np.array([2500, 0.4, 5])]

        self.proposal = WF_proposal(0.01, prior_distrs, prior_distr_args)
        self.proposal_args = []

    def get_theta_init(self):
        n = 1

        theta_init = self.prior.rvs(*self.prior_args)
        s_init = self.statistics(self.simulator(theta_init))

        while s_init[0] < 50 or s_init[1] < 50 or s_init[2] < 50:
            theta_init = self.prior.rvs(*self.prior_args)
            s_init = self.statistics(self.simulator(theta_init))
            n += 1

            if n % 100 == 0:
                print n, s_init
        print 'tried n = %d times for init' % (n)
        # return np.array([1700, 0.6, 3, 300, 140, 200])
        return theta_init.ravel()

    def statistics(self, vals):
        return cell_stats(vals)

    def simulator(self, theta):
        theta_vec = []
        index1 = 0

        for i, a in enumerate(['N', 'k', 'p1']):
            if a in self.fixed:
                theta_vec.append(self.true_true_args[i])
            else:
                theta_vec.append(theta[index1])
                index1 += 1

        return simulation(self.M, self.T,
                          float(theta_vec[0]),
                          float(theta_vec[1]),
                          float(theta_vec[2]),
                          self.x0, self.x1, self.x2)


if __name__ == "__main__":
    cell_names = ["SC", "TAC", "TDC"]
    T = 500
    N = 2000.0
    p1 = 0.1
    k = 3
    x0 = 1000
    x1 = 1000
    x2 = 1000
    # params={'p1': 0.59, 'T': 500, 'x2': 1000,
    #'k': 8.2, 'x1': 1000, 'M': 50, 'x0': 1000, 'N': 981}
    cells = simulate_all(T, N, k, p1, x0, x1, x2)
    print cell_stats(cells)
    pp.figure(figsize=(10, 5))
    pp.plot(cells / N, lw=2)
    pp.legend(cell_names)
    pp.xlabel("T")
    pp.ylabel("cell population")

    # M = 5
#   params = {"M":M,"T":T,"N":N,"k":k,"p1":p1,"x0":x0,"x1":x1,"x2":x2}
#   for i in range(10):
#     Y = simulation( params )
#     S = cell_stats( Y )

    # pp.figure( figsize=(7,9));
   #  for si in range(3):
   #    pp.subplot( 3,1,si+1)
   #    pp.hist( Y[:,si], 20, normed=True,color='b',alpha=0.5)
   #    ax = pp.axis()
   #    x = np.linspace( ax[0], ax[1], 100)
   #    pp.plot( x, gaussian_pdf( x, S[si],np.sqrt(S[si+3])), 'r-', lw=2 )
   #    pp.ylabel( cell_names[si])

    pp.show()
