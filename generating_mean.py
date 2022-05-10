import numpy as np
from anesthetic import MCMCSamples
import os
if not os.path.isdir('base'):
    chains = 'https://pla.esac.esa.int/pla/aio/product-action?COSMOLOGY.FILE_ID=COM_CosmoParams_base-plikHM-TTTEEE-lowl-lowE_R3.00.zip'

    import urllib.request

    urllib.request.urlretrieve(chains, "chains.zip")

    import zipfile

    with zipfile.ZipFile("chains.zip", 'r') as zip_ref:
        zip_ref.extractall(".")

    os.remove("chains.zip")
root = 'base/plikHM_TTTEEE_lowl_lowE_lensing/base_plikHM_TTTEEE_lowl_lowE_lensing'
planck_samples = MCMCSamples(root=root)
mu = planck_samples[paramnames].mean().values
Sig = planck_samples[paramnames].cov().values
invSig = np.linalg.inv(Sig[:-1, :-1])

bounds = np.array([planck_samples.limits[p] for p in paramnames], dtype=float)