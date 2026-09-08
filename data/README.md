# Data

Raw and processed data files are not tracked in git (see `.gitignore`).

## Norman et al. Perturb-seq (K562, GSE133344)

Download the harmonized AnnData version from scPerturb:

https://www.sanderlab.org/scPerturb/

Place the file here as:

```
data/norman.h5ad
```

`src/data/preprocessing.py` expects this path by default (see `configs/model.yaml`).
