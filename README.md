# estaudel - Nested Darwinian population simulation

## Content:

- `/estaudel/` : is the python library
- `/estaudel/stochastic` : Implement a stochastic birht-death-process with mutation for particles.
- `/estaudel/escafolding` : Implement collective level events, like collective reproduction and selection.
- `/estaudel/heredity`: Implement the red/blue model used in the heredity project.
- `/evolution_of_heredity/` : contains code for the figure in the
  "Evolutionary origins of heredity during major egalitarian
  transition in individuality" paper.
- `/evolution_of_heredity/full_trajectory.py`: shows how to use the bits
  and pieces from the library and do a full simulation.

## Getting started

Here is a quick way of setup a development environment, independant from your python install.

```
# 1. Create a folder to store everything.
mkdir nested_evolution
cd nested_evolution

# 2. Get the source
git clone git@gitlab.com:ecoevomath/estaudel.git

# 3. Optional: Create a virtual environment
# Use this if you do not want to mess with the python package installed on your computer.
# You do not need root access, but you will need the virtualenv package
# If you want to install on your main python environment, you can safely jump to step 4.
virtualenv env # `sudo pip3 install virtualenv` will install the package if it is missing.
source env/bin/activate  # Switch to the virtual environment

# 4. Install the python package in "developer mode".
pip3 install -e estaudel/

# 5. Try it out.
python3 estaudel/evolution_of_heredity/full_trajectory.py --N 10 --D 10
# or look at the jupyter notebook "evolution_of_heredity/getting_started.ipynb"
```
