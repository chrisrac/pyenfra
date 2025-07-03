# pyenfra
## Python Package for Environmental Fractal Analysis and Chaos Estimation
![Static Badge](https://img.shields.io/badge/release-v0.4-red?style=flat)
![Static Badge](https://img.shields.io/badge/pypi-comming_soon-orange?style=flat)
![Static Badge](https://img.shields.io/badge/python-v3.7%2B-blue?style=flat)
![Static Badge](https://img.shields.io/badge/license-MIT-green?style=flat)
![Static Badge](https://img.shields.io/badge/NOAA-NA19NOS4730207-navy?style=flat)

Official repository: [github: pyenfra](https://github.com/chrisrac/pyenfra)

## Description


pyenfra is a Python library for fractal analysis, modeling and chaos estimation in time-series with the emphasis on environmental datasets.
Package contains a suite of fractal and chaos metrics:
- Hurst exponent for rescaled range, 
- Detrended Fluctuation Analysis (DFA), 
- Multifractality by Generalized Hurst Slope, 
- Wavelet Transform Modulus Maxima (with varying bands and modulus methods), 
- Sample Entropy, 
- Recurrence Quantification Analysis (RQA), 
- Lyapunov exponents. 


## Installation


pyenfra can be installed using **pip** or locally by downloading package copy. 

pip install:
```python
pip install pyenfra
```
local:

use repositiory to obtain package copy.


## Usage


Below are a couple examples of package usage. 

Please refer to [examples.py](https://github.com/chrisrac/pyenfra/blob/main/examples/examples.py) for extended, detailed examples and computation workflows.

```python
import numpy as np
import matplotlib.pyplot as plt

import pyenfra

# Generate White noise sample data
ts_white = np.random.RandomState(0).randn(2000)

# Example: Compute Hurst exponent
h_value = pyenfra.functions.hurst(ts_white, num=30, min_n=10, min_segments=10)

# Example: Interpret Hurst
print(pyenfra.interpreters.interpret_hurst(ts_white, use_confidence_interval=False))

# Example: Plot Hurst climacogram for AR(1)
ax_hurst = pyenfra.plotting.plot_hurst(ts_white, num=30, min_n=10, min_segments=10,
                               figsize=(5,4), scatter_kwargs={'color':'C0'}, line_kwargs={'color':'C1'})
ax_hurst.figure.suptitle("Climacogram: AR(1) Persistent Process")
plt.show()

# Example: Compute Lyapunov Exponent
lyap_val, divergence, times = pyenfra.functions.lyapunov(ts_white, dim=3, tau=1, fs=1.0, max_iter=200, theiler=1)
print(f"Estimated Lyapunov exponent (logistic r=3.99): {lyap_val:.4f}")
print("Interpretation:", pyenfra.interpreters.interpret_lyapunov(lyap_val))
```


## Roadmap
Future works on the package include:
- [ ] HOST model integration.
- [ ] On demand functions.
  

## Contributing

Pull requests are welcome. 

For major changes, please open an issue first to discuss implementation or changes.


## Acknowledgment

This work was supported by NOAA grant NA19NOS4730207. Funding agency had no impact on work structure or findings.


## License
This package is available under [MIT](https://choosealicense.com/licenses/mit/) license.
