# tc-hard
This repository investigates TCR-peptide/-pMHC binding prediction on unseen peptides using state-of-the-art binding predictors.
The notebooks used to create the TChard dataset are included in `notebooks/notebooks.dataset/`.
The TChard dataset is available at: https://doi.org/10.5281/zenodo.6962043


# Content
```
tc-hard
│   README.md
│   ... 
│     
└───notebooks
│   │   notebooks.classification/ (TCR-peptide/-pMHC experiments)
│   │   notebooks.classification.results/ (plotting results of NetTCR2.0 and ERGO II)
│   │   notebooks.dataset/ (creation of the TChard dataset)   
│   
└───scripts/ (experiments, it mainly mirrors the content of notebooks.classification/)
│   
└───tcrmodels/ (Python package which wraps SOTA ML-based TCR models)
```

# tcrmodels
`tcrmodels` wraps deep learning TCR prediction models.
It includes:
* [ERGO II](https://github.com/IdoSpringer/ERGO-II)
* [NetTCR2.0](https://github.com/mnielLab/NetTCR-2.0)

## Install `tcrmodels`
```
cd tcrmodels
pip install .
pip install torch===1.4.0 torchvision===0.5.0 -f https://download.pytorch.org/whl/torch_stable.html
```

`tcrmodels` requires Python 3.6

## References
### ERGO II
Springer I, Tickotsky N and Louzoun Y (2021), Contribution of T Cell Receptor Alpha and Beta CDR3, MHC Typing, V and J Genes to Peptide Binding Prediction. Front. Immunol. 12:664514. DOI: https://doi.org/10.3389/fimmu.2021.664514

### NetTCR-2.0
Montemurro, A., Schuster, V., Povlsen, H.R. et al. NetTCR-2.0 enables accurate prediction of TCR-peptide binding by using paired TCRα and β sequence data. Commun Biol 4, 1060 (2021). DOI: https://doi.org/10.1038/s42003-021-02610-3

# License
For the content of this repositoy, we provide a non-commercial license, see LICENSE.txt
