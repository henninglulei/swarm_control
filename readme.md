main.py contains the code to run the simulation across the 100 iterations. Put the desired config into the list and run. It is saved as a dict in a binary npy file which can be displayed using the plot convergence notebook.


This notebook is also used to show the trajectories of single iterations. At the top of the notebook input your config and run the following cells.


A config consists of

- noise covariance: usually 0 matrix or given
- a filter object, use noop if unfiltered is desired
- a descriptor for later reference, string
- a timestep, usually given
- number of measurements per timestep T


There are 3 available filters

- NoOp -> does nothing -> unfiltered
- MaximumLikelihoodEstimator -> MLE
- RecursiveLeastSquares -> RLS/simple Kalman
