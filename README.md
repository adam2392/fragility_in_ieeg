# Neural Fragility In IEEG Data

A repository for generating figures for "Neural Fragility of iEEG as a marker for the seizure onset zone".
Notebooks use some helper functions from a private repository, but all figures to be generated for 
publication can be found in the `data/` directory.

For a demo of the neural fragility algorithm on a single subject, contact 
authors for the Gigantum demo.

Data Organization
-----------------

Data should be organized in the BIDS-iEEG format:

https://github.com/bids-standard/bids-specification/blob/master/src/04-modality-specific-files/04-intracranial-electroencephalography.md

System Requirements
===================
Generally to run the figure generation, one
simply needs a standard computer with enough RAM.
Minimally to generate the figures, probably a computer 
with 2GB RAM is sufficient.

We ran tests on computer with the following:

RAM: 16+ GB
CPU: 4+ cores, i7 or equivalent

Software: Mac OSX or Linux Ubuntu 18.04+. One should use Python3.6+.

Installation Guide
==================

Setup environment from pipenv. The `Pipfile` contains the Python 
libraries needed to run the figure generation in [notebook](neural_fragility_journal_figures.ipynb).

    pipenv install --dev

   # if dev versions are needed
    pipenv install https://api.github.com/repos/mne-tools/mne-bids/zipball/master --dev
    pipenv install https://api.github.com/repos/mne-tools/mne-python/zipball/master --dev

To generate a requirements.txt file for Gigantum, or pip installations:

    pipenv lock --requirements > requirements.txt

Demo
====
We can send a demo of fragility being run on a patient, (namely Patient_01 in paper).
The code and demo is not open-source due to license restrictions.
Please contact authors if interested.

Instructions for Use
====================
Run the notebook from beginning to end to generate figures, 
by pointing the path to the `data/` folder here.

