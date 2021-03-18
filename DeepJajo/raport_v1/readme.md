## Setup
```
conda env create -n bajojajo -f enviroment.yml 
conda activate bajojajo
```

## Dodanie env jako kernela jupytera
```
conda install -c anaconda ipykernel
python3 -m ipykernel install --user --name=bajojajo
```





## Do ewentualnego debugowania

```
conda install numpy scipy joblib scikit-learn --force-reinstall
```

```
conda install python-graphviz
CONDA_RESTORE_FREE_CHANNEL=1 conda env create --file enviroment.yml
```
