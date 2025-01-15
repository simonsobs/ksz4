# ksz4

ksz4 provides tools for estimating the kSZ trispectrum from CMB data, as motivated by Smith and Ferraro (https://arxiv.org/abs/1607.01769), Ferraro and Smith (https://arxiv.org/abs/1803.07036). Based on the code used for MacCrann et al. 2024 (https://arxiv.org/abs/2405.01188)

We use a quadratic estimator approach, i.e. the trispectrum, $C_L^{KK}$, is calculated as the power spectrum of $K = Q(T_1, T_2)$, with $Q(T_1, T_2)$ denoting a quadratic estimator applied to CMB temperature maps $T_1$ and $T_2$. We use the formalism because of the similiarities to CMB lensing estimation, from which we adapt various tools such as Gaussian noise ($N^0$) estimation and bias-hardening. 

As well as "lensing-hardened" estimators (see MacCrann et al. 2024), we provide estimators and analytic $N^0$ calcualations for the general case $C_L^{KK} = < Q(T_1,T_2) Q(T_3,T_4) >$, where $T_i$ are CMB temperature maps with different noise levels (e.g. because they have different foreground treatments applied). 

The notebooks `examples/test_signal_and_N0.ipynb` and `examples/test_bias-hardening.ipynb` give a fairly comprehensive demonstration of the usage of the code. 
