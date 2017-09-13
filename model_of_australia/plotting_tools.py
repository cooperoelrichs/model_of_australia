def density_plot(y, lims, color=None, linewidth=1):
    density = stats.gaussian_kde(y)
    x = np.linspace(lims[0], lims[1] , 1e5)
    plt.plot(x, density(x), color=color, linewidth=linewidth)

def pdf_plot(d, params, lims, color=None, linewidth=1):
    x = np.linspace(lims[0], lims[1], int(1e5))
    plt.plot(x, d.pdf(x, *params), color=color, linewidth=linewidth)

def add_legend(fig, ax, cm, quantiles):
    patches = [
        matplotlib.patches.Patch(
            color=cm((1-q)),
            label='P = %.2f' % q
        )
        for q in quantiles
    ]
    lg = plt.legend(handles=patches, loc=2, framealpha=1, frameon=True)  # facecolor='white'
    lg.get_frame().set_facecolor('white')
