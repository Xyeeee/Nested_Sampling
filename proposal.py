import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from getdist.mcsamples import MCSamplesFromCobaya
import getdist.plots as gdplt
from cobaya.run import run

plt.switch_backend("TkAgg")


def gauss(x, y, beta):
    # x here is the vectorized output of a hypothesis gaussian function with certain mean and sigma
    # the likelihood is computed based on the point wise gaussian likelihood on the entire input scale 0-2
    axis = np.linspace(0, 2, 100)
    z = stats.norm.pdf(axis, loc=x, scale=y)
    data = stats.norm.pdf(axis, loc=1, scale=0.4)
    diff = z - data
    likelihood = np.array([stats.norm.logpdf(d, loc=0, scale=0.1) for d in diff])
    return np.sum(likelihood)


info = {"likelihood": {"1dgaussian": {"external": gauss, "output_params": ["z"]}}, "params": {
    "x": {"prior": {"min": 0, "max": 2}, "ref": 0.5, "proposal": 0.01},
    "y": {"prior": {"min": 0, "max": 2}, "ref": 0.5, "proposal": 0.01},
    "beta": {"prior": {"min": 0, "max": 1}, "ref": 0.9, "proposal": 0.01}
}, "sampler": {"mcmc": {"Rminus1_stop": 1, "max_tries": 1000}},
        "prior": {"mixture": lambda beta, x: np.log(
            (1 - beta) * stats.uniform.pdf(x, loc=0, scale=2) + beta * stats.uniform.pdf(x, loc=0.9, scale=0.2))}}

updated_info, sampler = run(info)

gdsamples = MCSamplesFromCobaya(updated_info, sampler.products()["sample"])
gdplot = gdplt.get_subplot_plotter(width_inch=5)
gdplot.triangle_plot(gdsamples, ["x", "y", "beta"], filled=True)
plt.show()
