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
It is well advised to reduce the size of the validation set.
First ImageMagic also needs to be allowed to have more space on the harddrive.
```bash
sudo vim /etc/ImageMagick-6/policy.xml
# <policy domain="resource" name="disk" value="8GiB"/>
```
And than you can shrink the DS
```bash
mkdir -p data/scripts_test_small/img/
for IMG in $(ls data/scripts_test/img/*jpg); do convert $IMG -resize 1024x1024  data/scripts_test_small/img/$(basename $IMG); done
cp ./data/scripts_test/gt.csv ./data/scripts_test_small/
```

Finaly launch on the small validation dataset 
```bash
PYTHONPATH="./src/" python3 bin/train.py -device cuda -val_root ./data/scripts_test_small/
```



### Train Model
```bash
PYTHONPATH="./src" python3 ./bin/train.py 
```