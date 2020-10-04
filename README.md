# roufcp - Rough Fuzzy Changepoint Detection

Gradual Change-Point Detection Library based on Rough Fuzzy Changepoint Detection algorithm `roufcp`.

The package is available in [PyPI](https://pypi.org/project/roufcp/0.1/).

## Usage

```
>> import numpy as np
>> from roufcp import roufCP
>> X = np.concatenate([np.ones(20) * 5, np.zeros(20), np.ones(20) * 10]) + np.random.randn(60)
>> roufCP(delta = 3, w = 3).fit(X, moving_window = 10, k = 2)
```

Try `help(roufCP)` for detailed documentation.

`roufCP` is a class for Rough Fuzzy Changepoint Detection with the following attributes and functions.

* Attributes
    - `delta` : `int`, The fuzzyness parameter, typically between 5-100
    - `w` : `int`, The roughness parameter, typically between 5-100

* Methods
    - `fit_from_regularity_measure(X, regularity_measure, k)` :
        fit the data X with help of the regularity measure and output the estimated changepoints

    - `fit(X, moving_window, method, k)`: fit the data X with given regularity measures and output the estimated changepoints. The method argument defaults to kstest, available options are;
      - `meandiff` : Two sample mean difference
      - `ttest` : Two sample t test statistic
      - `kstest` : Two sample Kolmogorov test statistic
      - `mannwhitney` : Two sample Mann Whitney U statistic
      - `anderson-darling` : Two sample Anderson Darling test statistic
      - `adf` : Augmented Dickey Fuller test of stationarity with linear trend
      - `kpss` : Kwiatkowski–Phillips–Schmidt–Shin (KPSS) test of stationarity with linear trend 
    
    - `hypothesis_test(cp_list, cp_entropy, mu, sigma, a_delta)`:
        Performs hypothesis testing of the null hypothesis that there is no changepoint in the data, against the alternative that there is changepoint at the specified indices, and outputs the p-value
    


## Authors & Contributors

* Subhrajyoty Roy - https://subroy13.github.io/
* Ritwik Bhaduri - https://github.com/Ritwik-Bhaduri
* Sankar Kumar Pal - https://www.isical.ac.in/~sankar/


## License

This code is licensed under MIT License.
