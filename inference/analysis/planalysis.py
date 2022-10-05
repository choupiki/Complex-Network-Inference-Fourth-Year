import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
from scipy.stats import entropy

from inference import tools


def recon_error_nguyen(true_model, inferred_model):
    true_params = tools.triu_flat(true_model, k=0)
    inf_params = tools.triu_flat(inferred_model, k=0)
    numerator = np.sum((inf_params - true_params) ** 2)
    denominator = np.sum(true_params ** 2)
    return np.sqrt(numerator / denominator), numerator, denominator


def recon_error(true_model, inferred_model):
    # print(true_model.shape)
    true_params = tools.triu_flat(true_model, k=0)
    inf_params = tools.triu_flat(inferred_model, k=0)
    # excluding h at the moment, but why?
    # print(true_J.shape, inf_J.shape)
    # nParameters = true_params.size
    # print(nParameters)
    abs_diff = abs(true_params - inf_params)
    absError_perParam = np.mean(abs_diff)
    return absError_perParam
    # numerator = np.sum((inf_params - true_params) ** 2)
    # denominator = np.sum(true_params ** 2)
    # return # np.sqrt(numerator / denominator)


def KL_div(true_model, inferred_model, nbins=50):
    true_params = tools.triu_flat(true_model)
    inf_params = tools.triu_flat(inferred_model)
    true_dist, _ = np.histogram(true_params, bins=nbins)
    inf_dist, _ = np.histogram(inf_params, bins=nbins)
    S = entropy(inf_dist, true_dist)
    return S


class ErrorAnalysis:
    def __init__(
            self,
            model_IN,
            model_OUT,
            showOverview=True, showDistributions=True):

        self.m_true = model_IN
        self.m_inf = model_OUT
        self.showOverview = showOverview
        self.showDistributions = showDistributions

    # subsample should be tupple: (starting_index, size)
    def overview(self, sub_sample=None):
        true_model = self.m_true
        inf_model = self.m_inf
        if sub_sample is not None:
            starting_index, size = sub_sample
            true_model = true_model[
                starting_index: starting_index + size,
                starting_index: starting_index + size]
            inf_model = inf_model[
                starting_index: starting_index + size,
                starting_index: starting_index + size]

        true_params = tools.triu_flat(true_model, k=0)
        inf_params = tools.triu_flat(inf_model, k=0)

        model_error = np.abs(inf_model - true_model)
        r2_val = r2_score(true_params, inf_params)

        max_value = np.max(true_params)
        min_value = np.min(true_params)

        fig, ax = plt.subplots(2, 2)
        ax = ax.ravel()
        ax[0].imshow(true_model, vmin=min_value, vmax=max_value)
        ax[0].set(title='True Model')
        ax[1].imshow(inf_model, vmin=min_value, vmax=max_value)
        ax[1].set(title='Inferred Model')
        ax[2].imshow(model_error, vmin=min_value, vmax=max_value)
        ax[2].set(title='Error Matrix')
        ax[3].axline(
            (-1, -1), (1, 1), marker=',', color='k', transform=ax[3].transAxes)
        ax[3].plot(
            true_params, inf_params,
            linestyle='None',
            label=r'$R^{2}=$' + '{:.3f}'.format(r2_val))
        ax[3].set(xlabel='True', ylabel='Inferred', title='Reconstruction')
        # plt.savefig("N64PD.pdf")
        plt.legend()
        plt.show()

    def histograms(self):
        true_model = self.m_true
        inf_model = self.m_inf
        true_Js = tools.triu_flat(true_model, k=1)
        inf_Js = tools.triu_flat(inf_model, k=1)
        error_Js = abs(true_Js - inf_Js)

        true_hs = np.diagonal(true_model)
        inf_hs = np.diagonal(inf_model)
        error_hs = (true_hs - inf_hs)

        fig, ax = plt.subplots(2, 2)

        nbins = 100
        logswitch = False

        ax[0, 0].hist(inf_Js, bins=nbins, weights=np.ones_like(inf_Js) / len(inf_Js), alpha=0.3, log=logswitch, label='OutputCouplings')
        ax[0, 0].hist(true_Js, bins=nbins, weights=np.ones_like(true_Js) / len(true_Js), log=logswitch, label='InputCouplings')
        ax[1, 0].hist(error_Js,  bins=nbins, weights=np.ones_like(error_Js) / len(error_Js), log=logswitch, label='Error')

        ax[0, 1].hist(true_hs, bins=nbins, weights=np.ones_like(true_hs) / len(true_hs), log=logswitch, label='InputCouplings')
        ax[0, 1].hist(inf_hs, bins=nbins, weights=np.ones_like(inf_hs) / len(inf_hs), alpha=0.3, log=logswitch, label='OutputCouplings')
        ax[1, 1].hist(error_hs,  bins=nbins, weights=np.ones_like(error_hs) / len(error_hs), log=logswitch, label='Error')

        ax[0, 0].legend()
        ax[1, 0].legend()
        ax[0, 1].legend()
        ax[1, 1].legend()
        plt.show()


def histograms(true_model, inf_model, sub_sample=None):

    if sub_sample is not None:
        starting_index, size = sub_sample
        true_model = true_model[
            starting_index: starting_index + size,
            starting_index: starting_index + size]
        inf_model = inf_model[
            starting_index: starting_index + size,
            starting_index: starting_index + size]

    hs_true, Jupper_true, Jlower_true = tools.split_diag(true_model)
    hs_inf, Jupper_inf, Jlower_inf = tools.split_diag(inf_model)
    # print(hs_true)
    # print(hs_inf)
    '''
    fig, ax = plt.subplots(1, 2)
    ax = ax.ravel()
    ax[0].hist(hs_inf, bins=20, alpha=0.5)
    ax[1].hist(Jupper_inf.ravel(), bins=100, alpha=0.5)
    ax[1].set(ylim=(0, 300))
    plt.show()
    '''
    return np.mean(hs_inf)
    # things are going the wrongness
    # why are the values just wrong for the hs?!?
    # need to check my code that generates these really!
    # somehow how I put h in my modesl is wrong I think!
    # cause it's h / T or something like that that I store in
    # my matrix
    '''
    fig, ax = plt.subplots(2, 1)
    ax = ax.ravel()
    ax[0].imshow(Jupper_true)
    ax[1].imshow(Jlower_true)
    plt.show()

    reconst = Jupper_true + Jlower_true
    np.fill_diagonal(reconst, hs_true)
    fig, ax = plt.subplots(2, 1)
    ax = ax.ravel()
    ax[0].imshow(reconst)
    ax[1].imshow(Jlower_true)
    plt.show()
    '''
    # true_params = tools.triu_flat(true_model)
    # inf_params = tools.triu_flat(inf_model)
    # lets split them into h_s, and Js?
