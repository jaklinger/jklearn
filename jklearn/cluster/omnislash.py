# TODO could implement parallel processing for large branches
# TODO could compile with cython
"""
omnislash
=========

Quick and rough top-down hierarchical segmentation of
very large, very high dimensional data.
"""

from sklearn.decomposition import PCA
import numpy as np


def mean_and_var(X):
    """Calculate the mean and variance of a distribution
    simulataneously, rather than calculating the mean
    twice using numpy's :obj:`var` and :obj:`mean`.
    Stolen from https://stackoverflow.com/a/19391264/1571593

    Args:
        X (:obj:`np.array`): Array over which to calculate mean
                             and variance.
    Returns:
        mean and variance of the input.
    """
    mean = X.mean()
    c = X - mean
    var = np.dot(c, c)/X.size
    return mean, var


def evaluate_score(sorted_X, cut, tight=False):
    """The objective function, with the loss defined as the
    the ratio of the sum of variances (on either side of the
    segmentation) with respect to the variance in the region around the cut.

    Args:
        sorted_X (:obj:`np.array`): A **sorted** array over which to evaluate
                                    the loss at value :obj:`cut`.
        cut (float): The value at which to calculate to evaluate the loss.
        tight (bool): Whether or not use a tighter region around the cut in the
                      denominator of the loss.
    """
    # Calculate the mean and variance on either side of the cut
    cut_idx = np.searchsorted(sorted_X, cut)
    mean_left, var_left = mean_and_var(sorted_X[:cut_idx])
    mean_right, var_right = mean_and_var(sorted_X[cut_idx:])
    if tight:
        mean_left = mean_left + np.sqrt(var_left)
        mean_right = mean_right - np.sqrt(var_right)
    # Calculate the variance in the region of the cut
    left_idx = np.searchsorted(sorted_X, mean_left)
    right_idx = np.searchsorted(sorted_X, mean_right)
    _, var_between = mean_and_var(sorted_X[left_idx:right_idx])
    # Calculate the loss
    return (var_left + var_right)/var_between


class Omnislash:
    """Hierarchical segmentation of data along a single principal component.
    This is repeated iteratively along to data on either side of the respective
    segmentation.

    Args:
        min_leaf_size (int): Minimum cluster size.
        evr_max (int): Maximum cumulative 'explained variance ratio' to
                       consider when scanning principal component axes.
                       For example, if the first 3 components have
                       explained variance ratios of (0.5, 0.4, 0.1) and
                       :obj:`evr_max = 0.9`, only the first two components
                       will be scanned for segmentation. Setting
                       :obj:`evr_max = -1` only
                       scans the first component for each segmentation.
        sample_space_size (int): Number of points to uniformally sample on
                                 each principal component.
        pca_kwargs (dict): Any parameters to pass to
                           :obj:`sklearn.decomposition.PCA`. Note: :obj:`copy`
                           is always set to :obj:`False`
                           and :obj:`iterated_power`
                           (which can be overwritten) is set to 3 by default.
    """

    def __init__(self, min_leaf_size, evr_max=0.75,
                 sample_space_size=100, **pca_kwargs):
        if "iterated_power" not in pca_kwargs:
            pca_kwargs["iterated_power"] = 3
        pca_kwargs["copy"] = False

        self.pca = PCA(**pca_kwargs)
        self.min_leaf_size = min_leaf_size
        self.evr_max = evr_max
        self.sample_space_size = sample_space_size
        self.slash_tree = None

    def fit(self, X, *args, **kwargs):
        """Train hierarchical clustering tree :obj:`slash_tree`.

        Args:
            X (:obj:`np.ndarray`): Input data.
        """
        args = (self.pca, self.evr_max, self.min_leaf_size,
                self.sample_space_size)
        initial_indexer = np.array([True]*len(X))
        self.slash_tree = Slash(level=0, indexer=initial_indexer)
        self.slash_tree.slash(X, *args)

    def predict(self, X, *args, **kwargs):
        """Extract labels for the input data
        from the pretrained hierarchical clustering tree :obj:`slash_tree`

        Args:
            X (:obj:`np.ndarray`): Input data.
        Returns:
            labels (:obj:`np.array`): Cluster labels.
        """
        raw_indexes = [i for i in range(len(X))]
        indexes = np.array(raw_indexes)
        clusters = self.slash_tree.generate_clusters(indexes)
        for label, cluster in enumerate(clusters):
            for idx in cluster:
                indexes[idx] = label
        return indexes

    def fit_predict(self, X, *args, **kwargs):
        """A wrapper around :obj:`fit` and :obj:`predict`.

        Args:
            X (:obj:`np.ndarray`): Input data.
        Returns:
            labels (:obj:`np.array`): Cluster labels.
        """
        self.fit(X, *args, **kwargs)
        return self.predict(X, *args, **kwargs)


class Slash:
    """A node on a tree, with two children. Each node represents
    a cut in the data hyperspace.

    Args:
        level (int): Integer (zero-indexed) indicating the depth of the node.
        indexer (:obj:`np.array`): A indexing array (Boolean) which indicates
                                   how to arrive at the current node from the
                                   parent node.
    """
    def __init__(self, level, indexer):
        self.level = level
        self.indexer = indexer
        self.left = None
        self.right = None

    def find_best_cut(self, X, pca, evr_max, sample_space_size):
        """
        Args:
            X (:obj:`np.ndarray`): Input data.
            pca: A :obj:`sklearn.decomposition.PCA` instance.
            evr_max: Maximum cumulative 'explained variance ratio' to
                     consider when scanning principal component axes.
                     For example, if the first 3 components have
                     explained variance ratios of (0.5, 0.4, 0.1) and
                     :obj:`evr_max = 0.9`, only the first two components
                     will be scanned for segmentation. Setting
                     :obj:`evr_max = -1` only
                     scans the first component for each segmentation.
            sample_space_size (int): Number of points to uniformally sample on
                     each principal component.
        Returns:
            Two indexing arrays indicating clusters the two children
            after segmentation.
        """
        # Parameters to keep track of
        total_evr = 0
        best_score = None
        best_cut = None
        best_axis = None
        # Transform to principal components
        _X = pca.fit_transform(X[self.indexer])
        # Iterate until evr_max is reached
        for axis, evr in enumerate(pca.explained_variance_ratio_):
            # Slice and sort the array
            _X_axis = _X[:, axis]
            _sorted_X = np.sort(_X_axis)
            # Cut off the tails of the distribution
            low = np.percentile(_sorted_X, 1)
            high = np.percentile(_sorted_X, 99)
            # Sample 100 points on the distribution
            for cut in np.linspace(low, high, sample_space_size):
                score = evaluate_score(_sorted_X, cut,
                                       tight=(self.level == 0))
                # Update if a better score is found
                if best_score is None or score < best_score:
                    best_score = score
                    best_cut = cut
                    best_axis = axis
            # Quit if required
            total_evr += evr
            if evr_max == -1 or total_evr > evr_max:
                break
        # Generate the cut
        left_cut = (_X[:, best_axis] <= best_cut).flatten()
        return left_cut, ~left_cut

    def slash(self, X, pca, evr_max, min_leaf_size, sample_space_size):
        """Recursive slashing of the data via child nodes.

        Args:
            X (:obj:`np.ndarray`): Input data.
            pca: A :obj:`sklearn.decomposition.PCA` instance.
            evr_max: Maximum cumulative 'explained variance ratio' to
                     consider when scanning principal component axes.
                     For example, if the first 3 components have
                     explained variance ratios of (0.5, 0.4, 0.1) and
                     :obj:`evr_max = 0.9`, only the first two components
                     will be scanned for segmentation. Setting
                     :obj:`evr_max = -1` only
                     scans the first component for each segmentation.
            min_leaf_size (int): Minimum clust size.
            sample_space_size (int): Number of points to uniformally sample on
                     each principal component.
        """

        # Don't bother finding a cut if min leaf size will be reached
        if self.indexer.sum()/2 <= min_leaf_size:
            return

        # Perform the splitting
        left_cut, right_cut = self.find_best_cut(X, pca, evr_max,
                                                 sample_space_size)
        if min(left_cut.sum(), right_cut.sum()) < min_leaf_size:
            return

        # Generate children
        self.left = Slash(level=self.level+1,
                          indexer=self.indexer[left_cut])
        self.right = Slash(level=self.level+1,
                           indexer=self.indexer[right_cut])
        self.indexer = self.indexer & left_cut

        # Continue slashing
        self.left.slash(X[left_cut], pca, evr_max,
                        min_leaf_size, sample_space_size)
        self.right.slash(X[right_cut], pca, evr_max,
                         min_leaf_size, sample_space_size)

    def generate_clusters(self, labels):
        """Recursively extract cluster labels.

        Args:
            labels (:obj:`np.array`): A list of indexes of equal length
                                      to the trained data.
        Returns:
            Predicted cluster labels.
        """
        # Check whether children exist
        if self.left is None:
            return [set(labels)]
        # If so, recurse with the subset given by the indexer
        right = self.right.generate_clusters(labels[~self.indexer])
        left = self.left.generate_clusters(labels[self.indexer])
        return right + left
