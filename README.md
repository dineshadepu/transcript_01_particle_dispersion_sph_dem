# Paper Title

This repository contains the code and manuscript for (fill the description).


## Installation

This requires pysph to be setup along with automan. See the
`requirements.txt`. To setup perform the following:

0. Setup a suitable Python distribution, using
   [edm](https://docs.enthought.com/edm/) or [conda](https://conda.io) or a
   [virtualenv](https://virtualenv.pypa.io/).

1. Clone this repository:
```
	$ git clone https://github.com/dineshadepu/01_solute_dissolution_sph.git
```

2. Run the following from your Python environment:
```
    $ cd 01_solute_dissolution_sph
    $ pip install -r requirements.txt
```


## Generating the results

The paper and the results are all automated using the
[automan](https://automan.readthedocs.io) package which should already be
installed as part of the above installation. This will perform all the
required simulations (this can take a while) and also generate all the plots
for the manuscript.

To use the automation code, do the following::

    $ python automate.py
    # or
    $ ./automate.py

By default the simulation outputs are in the ``outputs`` directory and the
final plots for the paper are in ``manuscript/figures``.


## Building the paper

The manuscript is written with LaTeX and if you have that installed you may do
the following:

```
$ cd manuscript
$ pdflatex paper.tex
$ bibtex paper
$ pdflatex paper.tex
$ pdflatex paper.tex
```
