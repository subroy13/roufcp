import warnings
import numpy as np 
from scipy.stats import ks_2samp, mannwhitneyu, anderson_ksamp
from statsmodels.tsa.stattools import adfuller, kpss
from scipy.stats import multivariate_normal, norm


class roufCP:
    """
    A class for Rough Fuzzy Changepoint Detection Algorithm roufCP

    ...
    
    Attributes
    -----------
    - delta : int
        The fuzzyness parameter, typically between 5-100

    - w : int
        The roughness parameter, typically between 5-100

    - method: str
        Type of regularity measure method used to detect the changepoint


    Call
    -----
    - roufCP(delta, w):
        create an object of class roufCP with the specified parameters.

    Methods
    --------
    - set_approx(n_time):
        creates upper and lower approximations of the fuzzy rough sets

    - regularity_measure(X, moving_window = None, test_method = 'kstest'):
        create a regularity measure from the given data X

    - fit_from_regularity_measure(X, regularity_measure, k):
        fit the data X with help of the regularity measure and output the estimated changepoints

    - fit(X, moving_window, method, k):
        fit the data X with given regularity measures and output the estimated changepoints

    - hypothesis_test(cp_list, cp_entropy, mu, sigma, a_delta):
        Performs hypothesis testing of the null hypothesis that there is no changepoint in the data, against the alternative that there is changepoint at the specified indices, and outputs the p-value
        

    """

    def __init__(self, delta, w):
        self.delta = delta
        self.w = w
        self.method = None

    def set_approx(self, n_time):
        """
        A function for computing lower and upper approximations of fuzzy rough partitions

        ...

        Arguments
        ----------
        - n_time: int
            The number of timepoints in the data

        Output
        --------
        A tuple (M_lower, M_upper) 
        - M_lower : numpy 2d array
            Row i corresponds to the lower approximation of the first partition if the changepoint is estimated to be at i.
        
        - M_upper : numpy 2d array
            Row i corresponds to the upper approximation of the first partition if the changepoint is estimated to be at i.        

        """
        M_upper = np.zeros((n_time, n_time))
        M_lower = np.ones((n_time, n_time))
        
        # lower approx
        for T in range(n_time):
            for t in range(n_time):
                if T < (t - self.delta):
                    M_lower[T, t] = 0
                elif T < (t + self.w):
                    M_lower[T, t] = 2 * ( (T + self.delta - t)/(2*(self.delta + self.w))  )**2
                elif T < (t + 2 * self.w + self.delta):
                    M_lower[T, t] = 1 - 2 * ( (t + 2 * self.w - T + self.delta)/(2*(self.delta + self.w)) )**2
                else:
                    M_lower[T, t] = 1
        
        # upper approx
        for T in range(n_time):
            for t in range(n_time):
                if T < (t - self.delta - 2 * self.w):
                    M_upper[T, t] = 0
                elif T < (t - self.w):
                    M_upper[T, t] = 2 * ( (T + self.delta - t + 2 * self.w)/(2*(self.delta + self.w))  )**2
                elif T < (t + self.delta):
                    M_upper[T, t] = 1 - 2 * ( (t - T + self.delta)/(2*(self.delta + self.w)) )**2
                else:
                    M_upper[T, t] = 1
        return (M_lower, M_upper)


    def regularity_measure(self, X, moving_window = None, method = 'kstest'):
        """
        A function to create regularity measure that can be used to detect a changepoint

        ...

        Arguments
        ---------
        - X : list or numpy array, required
            1D array (n_time, ) or 2D array (n_time, n_features)

        - moving_window : int
            The half length of the moving window which will be used to compute the regularity measure
            Defaults to 1/25-th of the study period n_time

        - method : str
            Method of creating regularity measure. 
            Defaults to 'kstest'
            Available options are:
                * meandiff - Two sample mean difference
                * ttest - Two sample t test statistic
                * kstest - Two sample Kolmogorov test statistic
                * mannwhitney - Two sample Mann Whitney U statistic
                * anderson-darling - Two sample Anderson Darling test statistic
                * adf - Augmented Dickey Fuller test of stationarity with linear trend
                * kpss - Kwiatkowski–Phillips–Schmidt–Shin (KPSS) test of stationarity with linear trend 

        Output
        --------
        regularity_measure : numpy 1d array
            Array containing the regularity measure values based on the data X

        """
        X = np.array(X)  # convert the list to np array
        if len(X.shape) != 1 and len(X.shape) != 2:
            raise ValueError("Input X must be a 1D or 2D array")
        elif len(X.shape) == 1:
            X = X[:, np.newaxis]

        n_time = X.shape[0]
        n_features = X.shape[1]
        
        if moving_window is None:
            moving_window = int(n_time / 25)
        
        regularity_measure = np.zeros(n_time)

        # different available regularity measure
        if method in ['meandiff', 'ttest', 'kstest', 'mannwhitney', 'anderson-darling', 'adf', 'kpss']:
            self.method = method
        else:
            raise NotImplementedError
        
        for i in range(1, n_time):
            samp1 = X[max(0, i-moving_window):i]
            samp2 = X[i:min(n_time, i+moving_window)]

            if method == 'meandiff':
                regularity_measure[i] = 1/(1 + np.linalg.norm( np.mean(samp1, axis = 0) - np.mean(samp2, axis = 0) ))
            elif method == 'ttest':
                meandiff = np.mean(samp1, axis = 0) - np.mean(samp2, axis = 0)
                sigma = np.cov(np.concatenate([samp1, samp2], axis = 0), rowvar=False).reshape(n_features, n_features)
                regularity_measure[i] = 1 / (1 + (np.linalg.solve(sigma, meandiff)*meandiff).sum() )
            elif method == 'kstest':
                tmp = np.zeros(n_features)
                for j in range(n_features):
                    tmp[j] = ks_2samp(samp1[:, j], samp2[:, j]).statistic
                regularity_measure[i] = 1 / (1 + np.linalg.norm(tmp))
            elif method == 'mannwhitney':
                tmp = np.zeros(n_features)
                for j in range(n_features):
                    tmp[j] = mannwhitneyu(samp1[:, j], samp2[:, j]).statistic
                regularity_measure[i] = np.linalg.norm(tmp) / ((n_features ** 0.5) * samp1.shape[0] * samp2.shape[0])
            elif method == 'anderson-darling':
                tmp = np.zeros(n_features)
                for j in range(n_features):
                    tmp[j] = anderson_ksamp([samp1[:, j], samp2[:, j]]).statistic
                regularity_measure[i] = 1 / (1 + np.linalg.norm(tmp))
            elif method == 'adf':
                tmp = np.zeros(n_features)
                allsamp = np.concatenate((samp1, samp2), axis = 0)
                for j in range(n_features):
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        tmp[j] = adfuller(allsamp[:, j], regression = 'ct')[0]
                regularity_measure[i] = 1 / (1 + np.linalg.norm(tmp))
            elif method == 'kpss':
                tmp = np.zeros(n_features)
                allsamp = np.concatenate((samp1, samp2), axis = 0)
                for j in range(n_features):
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        tmp[j] = kpss(allsamp[:, j], regression = 'ct')[0]
                regularity_measure[i] = 1 / (1 + np.linalg.norm(tmp))
        return regularity_measure


    def fit_from_regularity_measure(self, regularity_measure, k = 10):
        """
        A function to estimate the changepoints from the regularity measure

        Arguments
        ---------
        regularity_measure : list or 1d numpy array
            The regularity measure R(t) which measures the similarity between two segments before the time t and after the time t locally. 
            A low value of the measure indicates a possibility of occurence of changepoint. 

        k : int
            The number of neighbours to compare to define local minima of the entropy. Low value of k generally means high number of detected changepoints and vice-versa. Defaults to 10.

        Output
        -------
        A python dictionary with two fields:
        
        - cp: list
            Array of estimated changepoints, sorted by corresponding values of the entropy in increasing order. So, the first element of the array is the most significant changepoint in that aspect.

        - exponential_entropy: 1d numpy array
            Array of the obtained entropy
        """
        
        regularity_measure = np.array(regularity_measure)    # convert to numpy array
        if len(regularity_measure.shape) != 1:
            raise ValueError("Regularity measure must be 1D array")

        n_time = regularity_measure.shape[0]
        M_lower, M_upper = self.set_approx(n_time)

        rho_T = 1 - np.dot(M_lower, regularity_measure) / np.dot(M_upper, regularity_measure)
        rho_Tc = 1 - np.dot(1 - M_upper, regularity_measure) / np.dot(1 - M_lower, regularity_measure)
        exp_entropy = (1/2) * (rho_T * np.exp(1 - rho_T) + rho_Tc * np.exp(1 - rho_Tc) )
        
        # estimate the changepoints
        cps = []
        for i in np.arange(k+1,(len(exp_entropy)-k-1)) :
            if (exp_entropy[i] == min(exp_entropy[(i-k):(i+k)])):
                cps.append(i)

        cps = sorted(cps, key=lambda cp: exp_entropy[cp])
        return {'cp' : cps, 'exponential_entropy' : exp_entropy}
    

    def fit(self, X, moving_window = None, method = 'kstest', k = 10):
        """
        A function to estimate the changepoints from the data with available regularity measures

        Arguments
        ----------
        - X : list or numpy array, required
            1D array (n_time, ) or 2D array (n_time, n_features)

        - moving_window : int
            The half length of the moving window which will be used to compute the regularity measure
            Defaults to 1/25-th of the study period n_time

        - method : str
            Method of creating regularity measure. 
            Defaults to 'kstest'
            Available options are:
                * meandiff - Two sample mean difference
                * ttest - Two sample t test statistic
                * kstest - Two sample Kolmogorov test statistic
                * mannwhitney - Two sample Mann Whitney U statistic
                * anderson-darling - Two sample Anderson Darling test statistic
                * adf - Augmented Dickey Fuller test of stationarity with linear trend
                * kpss - Kwiatkowski–Phillips–Schmidt–Shin (KPSS) test of stationarity with linear trend 

        k : int
            The number of neighbours to compare to define local minima of the entropy. Low value of k generally means high number of detected changepoints and vice-versa. Defaults to 10.


        Output
        -------
        A python dictionary with two fields:
        
        - cp: list
            Array of estimated changepoints, sorted by corresponding values of the entropy in increasing order. So, the first element of the array is the most significant changepoint in that aspect.

        - exponential_entropy: 1d numpy array
            Array of the obtained entropy

        """
        reg_measure = self.regularity_measure(X, moving_window, method)
        result = self.fit_from_regularity_measure(reg_measure, k)
        return result

    def hypothesis_test(self, cp_list, cp_entropy, mu, sigma, a_delta):

        """
        A function to perform hypothesis testing on the estimated changepoints

        Arguments
        ---------
        cp_list : 1d list or numpy array
            The list of the estimated changepoint indices for which the testing is to be performed

        cp_entropy : 1d list orr numpy array
            An array containing the values of the exponential entropy at the position of given changepoints in cp_list, in the same order.

        mu : float
            The theoretical mean of regularity measure under the null hypothesis of no changepoint

        sigma : 2D numpy array
            A 2D numpy array or matrix of size (n_time, n_time), where the entry (i, j) corresponds to theoretical covariance between the value of regularity measure at 'i' and at 'j', under the null hypothesis of no changepoint
        
        a_delta : float
            The scaling factor dependent on the length of moving window used to calculate regularity measure, such that the asymptotic distribution of the regularity measure becomes normal distribution


        Output
        -------
        A dictionary containing two fields
        - Individual p value : 1d array
            Asymptotic p-value for the individual alternatives that each of the given indices of cp_list correspond to a changepoint

        - Joint p value: float
            Asymptotic p-value for the joint alternative that all of the given indices of cp_list are changepoints
        
        """

        if a_delta < 1e-10:
            raise ValueError("a_delta must be positive")

        cp_list = np.array(cp_list)
        cp_entropy = np.array(cp_entropy)

        if len(sigma.shape) != 2:
            raise ValueError("sigma must be 2d numpy array")
        elif sigma.shape[0] != sigma.shape[1]:
            raise ValueError("sigma must be a square matrix")
        else:
            n_time = sigma.shape[0]

        M_lower, M_upper = self.set_approx(n_time)  # get the approximations

        tmp1 = M_lower.sum(axis = 1)
        tmp2 = M_upper.sum(axis = 1)
        bs = tmp1[cp_list] / tmp2[cp_list]
        bs_comp = (n_time - tmp2[cp_list]) / (n_time - tmp1[cp_list])

        diff1 = np.multiply(M_upper[cp_list, :], tmp1) - np.multiply(M_lower[cp_list, :], tmp2)
        diff1 /= (mu * (tmp2[cp_list] ** 2))[:, np.newaxis]
        diff2 = np.multiply(1 - M_lower[cp_list, :], 1 - tmp2) - np.multiply(1 - M_upper[cp_list, :], 1 - tmp1)
        diff2 /= (mu * ((1 - tmp1[cp_list]) ** 2))[:, np.newaxis]

        A_mat = diff1 * (bs * np.exp(bs))[:, np.newaxis] + diff2 * (bs_comp * np.exp(bs_comp))[:, np.newaxis] 
        sigma_star = A_mat @ sigma @ np.transpose(A_mat)
        sigma_star = (sigma_star + np.transpose(sigma_star))/2   # symmetrize the matrix in case of numerical errors

        H_star = (1 - bs) * np.exp(bs) + (1 - bs_comp) * np.exp(bs_comp)

        # Generate Multivariate p-value
        mpval = multivariate_normal.cdf(x = cp_entropy, mean = H_star, cov = sigma_star / (a_delta ** 2), allow_singular=True)

        # Generate individual p-values
        z = a_delta * (cp_entropy - H_star) / (np.maximum(np.diag(sigma_star), 1e-5) ** 0.5)
        ipval = norm.cdf(z)

        return {'Individual p value': ipval, 'Joint p value': mpval}

