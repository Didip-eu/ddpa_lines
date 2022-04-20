### trainset ICDAR2017_CLaMM_task1_task3.zip

### Download train-set
```bash
mkdir -p data
(cd data; wget http://clamm.irht.cnrs.fr/wp-content/uploads/ICDAR2017_CLaMM_task1_task3.zip; unzip ICDAR2017_CLaMM_task1_task3.zip)
# removing non existing file that crashes dataset
cat data/ICDAR2017_CLaMM_task1_task3/@ICDAR2017_CLaMM_task1_task3.csv | grep -v 'Task_1_3;IRHT_P_009783.tif;5;11' > /tmp/gt.csv
cp /tmp/gt.csv data/ICDAR2017_CLaMM_task1_task3/@ICDAR2017_CLaMM_task1_task3.csv
```

### Download validation-set
```bash
mkdir -p data
(cd data; wget http://clamm.irht.cnrs.fr/wp-content/uploads/ICDAR2017_CLaMM_task1_task3.zip; unzip ICDAR2017_CLaMM_task1_task3.zip)
```

### Train Model
```bash
PYTHONPATH="./src" python3 ./bin/train.py 
```