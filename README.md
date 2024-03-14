
------------------------------------------------------------------------

# IntervalModelEstimator 

------------------------------------------------------------------------

**IntervalModelEstimator** is a Python tool designed for feature selection on spectroscopy or similar datasets. Inspired by the iPLS method pioneered by Norgard (refer to references), this tool extends the functionality of iPLS to accommodate a broader range of target variables, including both regression-based, dichotomous and multiclass chemometric analyses.

While iPLS traditionally relies on Partial Least Squares (PLS) regression, intervalModelEstimator offers the flexibility for users to specify their choice of classifier or regressor. This dynamicity enables the method to seamlessly adapt to various chemometric analysis tasks, whether they involve classification or regression.

It also includes plot that allow users to visualize the performance of each interval compared to the full spectrum.

------------------------------------------------------------------------

### Usage

------------------------------------------------------------------------

1.  Clone the repository:

        git clone https://github.com/habeeb3579/interval-model-estimator.git

2.  Open test.ipynb file and run

------------------------------------------------------------------------

### References

------------------------------------------------------------------------

[1] Nørgaard, L., Saudland, A., Wagner, J., Nielsen, J.P., Munck, L. and Engelsen, S.B., 2000. Interval partial least-squares regression (i PLS): A comparative chemometric study with an example from near-infrared spectroscopy. Applied spectroscopy, 54(3), pp.413-419.