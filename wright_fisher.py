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


def simulation(params):
    #params={'p1': 0.59, 'T': 500, 'x2': 1000,
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


class Wright_Fisher_Problem(ABC_Problem):
    def __init__(self):
        self.y_dim = 6
        self.y_star = np.array([10.205, 1782.825, 198.47, 8.3685592,
                                9.76456763, 5.81681082])

        self.prior = distr.uniform_nd
        self.prior_args = [np.array([10,   0, 1,  10,  10,  10]),
                           np.array([3000, 1, 20, 400, 400, 400])]

        self.proposal = distr.multivariate_normal
        self.proposal_args = [np.diag([400, 0.2, 5, 60, 60, 60])]

        self.use_log = False

        self.theta_init = self.prior.rvs(*self.prior_args)

        self.T = 300

    def statistics(self, vals):
        return cell_stats(vals)

    def simulator(self, theta):
        return simulate_all(self.T, theta[0], theta[1], theta[2],
                            theta[3], theta[4], theta[5])


if __name__ == "__main__":
    cell_names = ["SC", "TAC", "TDC"]
    T = 300
    N = 2000.0
    p1 = 0.4
    k = 5
    x0 = 100
    x1 = 300
    x2 = 100
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
