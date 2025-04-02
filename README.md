
# A2 Coursework
### Keying Song (ks2146)

This repository contains my submission for the A2 coursework, which consists of three modules: PET-CT Image Reconstruction, MRI Image Denoising and CT Image segmentation & classification.

## Declaration
No auto-generation tools were used in this coursework.

## Project Structure
The main structure of the package `a2cw` is like:
```
.
├── a2cw/
│   ├── __init__.py               # expose all classes and functions for importing
│   ├── classification.py         # module for Ex3.2
│   ├── denoising.py              # module for Ex2.2
│   ├── reconstruction.py         # module for Ex1.1~1.5
│   └── segmentation.py           # module for Ex3.1
|
├── report/                       # coursework report
|
├── data/                         # the data used
├── results/                      # the folder to save all the results
|
├── main/                         # the folder to directly answer the questions of coursework 
|
├── pyproject.toml                     
└── README.md               
```

## Installation

1. Clone the repository:
```bash
git clone https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/a2_coursework/ks2146.git
```

2. Install: In the root directory:
```bash
pip install .
```

3. Use:
- After installing, all the classes and functions in `a2cw` can be imported and used.
```python
from a2cw import os_sart, sirt
```
- Run the notebook files in the folder `main` to run all the experiments for each module.
