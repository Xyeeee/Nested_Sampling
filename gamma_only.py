# Adam Ormondroyd
# from mpi4py import MPI
import os
import matplotlib.pyplot as plt
import numpy as np
from anesthetic import MCMCSamples, NestedSamples
import polychord.PolyChordLite.pypolychord as pp
from polychord.PolyChordLite.pypolychord.settings import PolyChordSettings as Settings

plt.switch_backend("TkAgg")
if not os.path.isdir('base'):
    chains = 'https://pla.esac.esa.int/pla/aio/product-action?COSMOLOGY.FILE_ID=COM_CosmoParams_base-plikHM-TTTEEE-lowl-lowE_R3.00.zip'

    import urllib.request

    urllib.request.urlretrieve(chains, "chains.zip")

    import zipfile

    with zipfile.ZipFile("chains.zip", 'r') as zip_ref:
        zip_ref.extractall(".")

    os.remove("chains.zip")

DIM_NUM = 6

# Load the planck samples into MCMCSamples object
root = 'base/plikHM_TTTEEE_lowl_lowE_lensing/base_plikHM_TTTEEE_lowl_lowE_lensing'
planck_samples = MCMCSamples(root=root)
# planck_samples["gamma"] = np.random.beta(DIM_NUM, 1, (planck_samples.shape[0]))
planck_samples["gamma"] = np.random.uniform(1e-5, 1, (planck_samples.shape[0]))
planck_samples.limits["gamma"] = (1e-5, 1.)

# Define the parameters we're working with (up to the first 27 -- which are the 'true' rather than derived parameters)
paramnames = np.append(planck_samples.columns[:DIM_NUM], planck_samples.columns[-1])

mu = planck_samples[paramnames].mean().values
Sig = planck_samples[paramnames].cov().values
invSig = np.linalg.inv(Sig[:-1, :-1])

bounds = np.array([planck_samples.limits[p] for p in paramnames], dtype=float)
lower = bounds[:-1, 0]
upper = bounds[:-1, 1]
diff_og = upper - lower
vol_og = np.prod(diff_og)


# ------------------------------------------
def loglikelihood(theta):
    return -(theta - mu[:-1]) @ invSig @ (theta - mu[:-1]) / 2, []


def loglikelihood_tilde(theta_full):
    theta = theta_full[:-1]
    gamma = theta_full[-1]
    return loglikelihood(theta)[0] + np.log(pi(theta) / pi_tilde(theta, gamma)), []


def pi_tilde(theta, gamma):
    upper_new = mu[:-1] + (upper - mu[:-1]) * (1 - gamma)
    lower_new = mu[:-1] - (mu[:-1] - lower) * (1 - gamma)
    diff_new = upper_new - lower_new
    vol_new = np.prod(diff_new)
    result = ((theta < upper_new) & (theta > lower_new)).mean() / vol_new
    return result


def pi(theta):
    return ((theta < upper) & (theta > lower)).mean() / vol_og


def prior(cube_full):
    cube = cube_full[:-1]
    # beta = cube_full[-1]
    gamma = cube_full[-1]
    # Tightened prior bounds, will improve the convergence
    upper_new = mu[:-1] + (upper - mu[:-1]) * (1 - gamma)
    lower_new = mu[:-1] - (mu[:-1] - lower) * (1 - gamma)
    diff_new = upper_new - lower_new

    theta = cube * diff_new + lower_new

    theta_full = np.empty_like(cube_full)
    theta_full[:-1] = theta
    theta_full[-1] = gamma
    return theta_full


# ------------------------------------------
# mode: adding beta dynmically
# nDims = len(paramnames) + 1
# mode: adding beta statically

nDims = len(paramnames)
nDerived = 0
nlive = 50
settings = Settings(nDims, nDerived, file_root="default", nlive=nlive)
settings.read_resume = False
settings.write_resume = False

pp.run_polychord(loglikelihood_tilde, nDims, nDerived, settings, prior=prior)

poly_samples = NestedSamples(root="chains/default", columns=paramnames)
poly_samples.tex = planck_samples.tex

# Plotting
fig, ax = planck_samples.plot_2d(paramnames)
poly_samples.plot_2d(ax)

poly_samples.gui()
plt.show()
