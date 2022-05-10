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
planck_samples["delta"] = np.random.uniform(0., 1., (planck_samples.shape[0]))
planck_samples["beta"] = np.random.beta(1, DIM_NUM, (planck_samples.shape[0]))
# planck_samples["gamma"] = np.random.uniform(1e-5, 1., (planck_samples.shape[0]))
planck_samples.limits["delta"] = (0., 1.)
planck_samples.limits["beta"] = (0., 1.)

# Define the parameters we're working with (up to the first 27 -- which are the 'true' rather than derived parameters)
paramnames = np.append(planck_samples.columns[:DIM_NUM], planck_samples.columns[-2])
paramnames = np.append(paramnames, planck_samples.columns[-1])

mu = planck_samples[paramnames].mean().values
Sig = planck_samples[paramnames].cov().values
invSig = np.linalg.inv(Sig[:-2, :-2])

# implement bad prior center
bad_mu = mu - 3 * np.sqrt(np.diag(Sig))

bounds = np.array([planck_samples.limits[p] for p in paramnames], dtype=float)
lower = bounds[:-2, 0]
upper = bounds[:-2, 1]
diff_og = upper - lower
vol_og = np.prod(diff_og)


# ------------------------------------------
def loglikelihood(theta):
    return -(theta - mu[:-2]) @ invSig @ (theta - mu[:-2]) / 2, []


def loglikelihood_tilde(theta_full):
    theta = theta_full[:-2]
    beta = theta_full[-1]
    delta = theta_full[-2]
    return loglikelihood(theta)[0] + np.log(pi(theta) / pi_tilde(theta, beta, delta)), []


def pi_tilde(theta, beta, delta):
    upper_new = bad_mu[:-2] + (2 * delta) * 3 * np.sqrt(np.diag(Sig))[:-2]
    lower_new = bad_mu[:-2] - (2 - 2 * delta) * 3 * np.sqrt(np.diag(Sig))[:-2]
    diff_new = upper_new - lower_new
    vol_new = np.prod(diff_new)
    result = beta * ((theta < upper) & (theta > lower)).mean() / vol_og + (1 - beta) * (
            (theta < upper_new) & (theta > lower_new)).mean() / vol_new
    return result


def pi(theta):
    return ((theta < upper) & (theta > lower)).mean() / vol_og


def prior(cube_full):
    cube = cube_full[:-2]
    beta = cube_full[-1]
    delta = cube_full[-2]
    upper_new = bad_mu[:-2] + (2 * delta) * 3 * np.sqrt(np.diag(Sig))[:-2]
    lower_new = bad_mu[:-2] - (2 - 2 * delta) * 3 * np.sqrt(np.diag(Sig))[:-2]
    diff_new = upper_new - lower_new
    x_1 = np.zeros(len(paramnames) - 2)
    x_2 = (beta * (lower_new - lower) / diff_og)
    x_3 = ((1 - beta) + beta * (upper_new - lower) / diff_og)
    x_4 = np.ones(len(paramnames) - 2)

    if beta == 0:
        theta = cube * diff_new + lower_new
    else:
        theta = ((x_1 <= cube) & (cube < x_2)) * (cube * diff_og / beta + lower) + \
                ((x_2 <= cube) & (cube < x_3)) * (cube + lower_new * (beta / diff_og + (1 - beta) / diff_new) - beta * (
                lower_new - lower) / diff_og) / (beta / diff_og + (1 - beta) / diff_new) + \
                ((x_3 <= cube) & (cube <= x_4)) * ((cube - (1 - beta)) * diff_og / beta + lower)

    theta_full = np.empty_like(cube_full)
    theta_full[:-2] = theta
    theta_full[-2] = delta
    theta_full[-1] = beta
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

# pp.run_polychord(loglikelihood, nDims, nDerived, settings, prior=prior)
pp.run_polychord(loglikelihood_tilde, nDims, nDerived, settings, prior=prior)

poly_samples = NestedSamples(root="chains/default", columns=paramnames)
poly_samples.tex = planck_samples.tex

# paramnames = np.append(paramnames, 'beta')
# Plotting
fig, ax = planck_samples.plot_2d(paramnames)
poly_samples.plot_2d(ax)

poly_samples.gui()
plt.show()
