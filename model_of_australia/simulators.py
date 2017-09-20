import numpy as np
from scipy import stats


class GDPSimulator():
    def fit_distribution(X, dist):
        return dist.fit(X[np.isfinite(X)])

    def non_standard_cauchy(a, b, size=None):
        return np.random.standard_cauchy(b, size=size) + a

    def run(gdp_d_data, gdp_data, distribution, n_years, n_iter):
        n_iter = int(n_iter)
        params = GDPSimulator.fit_distribution(gdp_d_data, distribution['fit'])
        init_gdp = gdp_data[-1]
        return GDPSimulator.simulate(init_gdp, n_years, n_iter, params, distribution['sim'])

    def simulate(init_gdp, n_years, n_iter, dist_params, dist_simulator):
        gdp_sim = np.empty((n_iter, n_years))
        for i in range(n_iter):
            delta_gva = dist_simulator(*dist_params, size=n_years)
            ratio_gdp = np.cumprod(delta_gva + 1)
            gdp_sim[i, :] = init_gdp * ratio_gdp

        return gdp_sim


class SimpleSimulator():
    def run(gdp_data, params, n_years, n_iter):
        n_iter = int(n_iter)

        init_gdp = gdp_data[-1]
        simulation = SimpleSimulator.simulate(
            init_gdp, n_years, n_iter, params
        )
        return simulation

    def simulate(init_gdp, n_years, n_iter, params):
        gdp_sim = np.empty((n_iter, n_years))
        for i in range(n_iter):
            mu = params['mu'][0][2].rvs(*params['mu'][0][1])
            sd = params['sd'][0][2].rvs(*params['sd'][0][1])
            deltas = stats.norm.rvs(mu, sd, size=n_years)

            ratios = np.cumprod(deltas + 1)
            gdp_sim[i, :] = init_gdp * ratios
        return gdp_sim



class SharedVarianceInternationalGDPSimulator():
    def run(gdp_data, params, n_years, n_iter):
        n_iter = int(n_iter)

        init_gdp = gdp_data[-1]
        simulation = SharedVarianceInternationalGDPSimulator.simulate(
            init_gdp, n_years, n_iter, params
        )
        return simulation


    def simulate(init_gdp, n_years, n_iter, params):
        gdp_sim = np.empty((n_iter, n_years))
        for i in range(n_iter):
            mu = params['mu_australia'][0][2].rvs(*params['mu_australia'][0][1])
            sd = params['sd'][0][2].rvs(*params['sd'][0][1])
            deltas = stats.norm.rvs(mu, sd, size=n_years)

            ratios = np.cumprod(deltas + 1)
            gdp_sim[i, :] = init_gdp * ratios

        return gdp_sim


class GDPSimulatorWithCorrelatedSectors():
    # Zero Means test.
    def run(gva_data, params, n_years, n_iter):
        n_iter = int(n_iter)

        init_gva = gva_data[-1, :]
        simulation = GDPSimulatorWithCorrelatedSectors.simulate(
            init_gva, n_years, n_iter, params
        )
        return simulation

    def simulate(init_gva, n_years, n_iter, params):
        gdp_sim = np.empty((n_iter, n_years))
        gdps_sim = np.empty((n_iter, n_years))

        for i in range(n_iter):
            D_gva = np.zeros((n_years, init_gva.shape[0]))
            V_mu_eco = params['mu_eco'][0][2].rvs(*params['mu_eco'][0][1], size=n_years)

            for j, sd_gva_j_spec in enumerate(params['SD_gva']):
                _, sd_gva_j_betas, sd_gva_j_dist = sd_gva_j_spec
                sd_gva_j = sd_gva_j_dist.rvs(*sd_gva_j_betas)
                D_gva[:, j] = stats.norm.rvs(V_mu_eco, sd_gva_j, size=n_years)

            gva_sim_i = np.empty((n_years, D_gva.shape[1]))
            for y in range(n_years):
                if y == 0:
                    gva_sim_i[y, :] = init_gva * (1 + D_gva[y, :])
                elif y > 0:
                    gva_sim_i[y, :] = gva_sim_i[y-1, :] * (1 + D_gva[y, :])

            gdp_sim[i, :] = gva_sim_i.sum(axis=1)

        return gdp_sim
