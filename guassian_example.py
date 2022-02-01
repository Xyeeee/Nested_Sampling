import numpy as np

# First generate the actual data D
mu, sigma, D_size = 0.3, 0.45, 1000
Data = np.random.normal(mu, sigma, D_size)


# Define the likelihood function P(D|theta, model)
def L(D, theta):
    return np.product(
        [np.exp(-np.power(d - theta[0], 2.) / (2 * np.power(theta[1], 2.)) * np.sqrt(2 * np.pi)) for d in D ]
    )

    # a = np.exp(np.sum([-np.power(d - theta[0], 2.) / (2 * np.power(theta[1], 2.)) for d in D]))
    # b = np.power(
    #     1 / (theta[1] * np.sqrt(2 * np.pi)), D_size)
    # return a / b

# Define Prior pi(theta), here theta 2 dimensional, assume within range of unity
N_sample = 200  # Number of samples of theta to be drawn in prior reservoir
mu_sample = np.random.uniform(1e-5, 1, N_sample)
sigma_sample = np.random.uniform(1e-5, 1, N_sample)
prior = np.vstack((mu_sample, sigma_sample)).T
# prior = [(mu_sample[i], sigma_sample[i]) for i in range(N_sample)]
mask = np.arange(N_sample)

# Main loop of nested sampling algorithm
J = 20  # Termination number of iteration steps
N = 5  # Number of samples from prior kept at each step
sample_numbers = np.random.choice(mask, N, replace=False)
prior_sample = [prior[i] for i in sample_numbers]
np.delete(mask, sample_numbers)
Z, X_0 = 0, 1
L_list = []
for i in range(1, J + 1):
    L_list = [L(Data, theta_sample) for theta_sample in prior_sample]
    L_min, theta_min_index = np.amin(L_list), np.argmin(L_list)
    X_i = np.exp(-i / N)
    w_i = X_i - X_0
    X_0 = X_i
    Z += L_min * w_i
    # Now need to replace the L_min element with another element with a higher likelihood
    new_point = np.random.choice(mask, 1)[0]
    while L(Data, prior[new_point]) <= L_min:
        new_point = np.random.choice(mask, 1)[0]
    np.delete(mask, new_point)
    prior_sample[theta_min_index] = prior[new_point]
Z += np.sum(L_list)*X_0/N

