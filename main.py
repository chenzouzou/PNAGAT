import warnings

from train import Train
from utilsnew import plot_auc_curves, plot_prc_curves


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    fprs, tprs, auc, precisions, recalls, prc = Train(directory='data',
                                                      epochs=1000,
                                                      attn_size=64,
                                                      attn_heads=8,
                                                      out_dim=512,
                                                      dropout=0.5,
                                                      slope=0.2,
                                                      lr=0.001,
                                                      wd=5e-3,
                                                      random_seed=11,
                                                      cuda=True,
                                                      model_type='PNAGAT')  

    plot_auc_curves(fprs, tprs, auc, directory='roc_result', name='test_auc')
    plot_prc_curves(precisions, recalls, prc, directory='roc_result', name='test_prc')
