# The SIR Model

This repository contains the code for the blog post on [The SIR model: A mathematical approach to epidemic dynamics](https://www.fabriziomusacchio.com/blog/2020-12-11-sir_model/). For further details, please refer to this  post.

For reproducibility:

```powershell
conda create -n sir_model_covid19 -y python=3.9
conda activate sir_model_covid19
conda install -y mamba
mamba install -y pandas matplotlib numpy scipy scikit-learn ipykernel notebook ipympl mplcursors
```

## Acknowledgement
I acknowledge that the main code is based on this [blog post](https://numbersandshapes.net/posts/fitting_sir_to_data_in_python/)  and this [documentation page](https://scientific-python.readthedocs.io/en/latest/notebooks_rst/3_Ordinary_Differential_Equations/02_Examples/Epidemic_model_SIR.html). I have made some modifications to the code to make it more readable and understandable.