from .plot_2d_separator import plot_2d_separator
from .plot_kneighbors_regularization import plot_kneighbors_regularization, \
    plot_regression_datasets, make_dataset
from .plot_linear_svc_regularization import plot_linear_svc_regularization
from .plot_interactive_tree import plot_tree_interactive
from .plot_interactive_forest import plot_forest_interactive
from .plot_rbf_svm_parameters import plot_rbf_svm_parameters
from .plot_rbf_svm_parameters import plot_svm_interactive
from .plot_scaling import plot_scaling, plot_relative_scaling
from .plot_digits_dataset import digits_plot
from .plot_pca import plot_pca_illustration

__all__ = ['plot_2d_separator', 'plot_kneighbors_regularization',
           'plot_linear_svc_regularization', 'plot_tree_interactive',
           'plot_regression_datasets', 'make_dataset',
           "plot_forest_interactive", "plot_rbf_svm_parameters",
           "plot_svm_interactive", 'plot_scaling', 'digits_plot',
           'plot_relative_scaling', 'plot_pca_illustration']
