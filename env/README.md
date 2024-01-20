# Python Environment

The conda environment use python version 3.11.

- Performance improvements: Python 3.11 was expected to have faster interpreter startup times and improved performance for certain types of operations, such as list and dictionary manipulation.

- Improved error messages: Python 3.11 was expected to have more informative and user-friendly error messages for common mistakes.

- Structural pattern matching: A new feature called structural pattern matching was planned for Python 3.11. This would allow developers to match complex patterns in data structures using a syntax similar to the switch statement in other programming languages.

- Additional standard library modules: Several new modules were planned to be added to the Python standard library in Python 3.11, including the graphlib module for working with graph structures and the zones module for handling time zones.

- Changes to typing annotations: The typing annotations in Python 3.11 were expected to be simplified and made more consistent with the rest of the language.

## Install scripts

```{bash}
# Create a conda environment from conda environment.yml file
conda create --name newenv --file environment.yml

# e.g
conda create --name mds-g36 --file environment.yml

# After installation, activate conda env via conda activate
conda activate mds-g36

# install required python dependencies
pip3 install -r requirements.txt
```

## Pre-build packages

In the environment.yml and requirements.txt, including:

- Jupyter, Jupyter Lab
- Matplotlib, Seaborn, Plotly, Folium, hvplot
- Pandas, Numpy, Geopandas, Polars
- Pytorch, Scikit-learn, statsmodel
