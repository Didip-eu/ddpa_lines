Finding the images for the old groundtruth
```bash
(mkdir -p data;cd data; cp -Rp /mnt/y/data/projekte/DiDip/MOM-data/ganzeUrkunde/ ./)
(mkdir -p /tmp/icarus;sshfs becore@images.icar-us.eu:/opt/tank/images /tmp/icarus  -o ro)
mkdir -p data/images

for FNAME in $(echo "doc('./data/ganzeUrkunde/104_ganzeUrkunden_Geras.xml')/annotation/file/object/bbox/concat(../../@filename,codepoints-to-string(10))" | saxonb-xquery - | tail -c +39); do cp /tmp/icarus/monasterium/pics/104/$FNAME data/images; done

for FNAME in $(echo "doc('./data/ganzeUrkunde/106_ganzeUrkunde.xml')/annotation/file/object/bbox/concat(../../@filename,codepoints-to-string(10))" | saxonb-xquery - | tail -c +39); do cp /tmp/icarus/monasterium/pics/106/$FNAME ./data/images/ ; done

for FNAME in $(echo "doc('./data/ganzeUrkunde/107_ganzeUrkunden.xml')/annotation/file/object/bbox/concat(../../@filename,codepoints-to-string(10))" | saxonb-xquery - | tail -c +39); do cp /tmp/icarus/monasterium/pics/107/$FNAME ./data/images/  ; done

or FNAME in $(echo "doc('./data/ganzeUrkunde/108_ganzeUrkunden.xml')/annotation/file/object/bbox/concat(../../@filename,codepoints-to-string(10))" | saxonb-xquery - | tail -c +39); do cp /tmp/icarus/monasterium/pics/108/$FNAME ./data/images/  ; done

```

Stuck on the following place:
Only half the images work because a space in the filename.
```bash
or FNAME in $(echo "doc('./data/ganzeUrkunde/114_ganzeUrkunden.xml')/annotation/file/object/bbox/concat(../../@filename,codepoints-to-string(10))" | saxonb-xquery - | tail -c +39); do cp /tmp/icarus/monasterium/pics/114/$FNAME ./data/images/  ; done

```
Solution:
```bash
IFS=$'\n'
for FNAME in $(echo "doc('./data/ganzeUrkunde/114_ganzeUrkunden.xml')/annotation/file/object/bbox/concat(../../@filename,codepoints-to-string(10))" | saxonb-xquery - | tail -c +39| dos2unix|sed 's/^ *//g'|sed 's/ /\\\ /g'); do cp /tmp/icarus/monasterium/pics/114/$FNAME ./data/images/ ; done

for FNAME in $(echo "doc('./data/ganzeUrkunde/119_ganzeUrkunden.xml')/annotation/file/object/bbox/concat(../../@filename,codepoints-to-string(10))" | saxonb-xquery - | tail -c +39| dos2unix|sed 's/^ *//g'|sed 's/ /\\\ /g'); do cp /tmp/icarus/monasterium/pics/119/$FNAME ./data/images/ ; done

for FNAME in $(echo "doc('./data/ganzeUrkunde/124_ganzeUrkunden.xml')/annotation/file/object/bbox/concat(../../@filename,codepoints-to-string(10))" | saxonb-xquery - | tail -c +39| dos2unix|sed 's/^ *//g'|sed 's/ /\\\ /g'); do cp /tmp/icarus/monasterium/pics/124/$FNAME ./data/images/ ; done

for FNAME in $(echo "doc('./data/ganzeUrkunde/132_ganzeUrkunden_Seitenstetten.xml')/annotation/file/object/bbox/concat(../../@filename,codepoints-to-string(10))" | saxonb-xquery - | tail -c +39| dos2unix|sed 's/^ *//g'|sed 's/ /\\\ /g'); do cp /tmp/icarus/monasterium/pics/132/$FNAME ./data/images/ ; done

for FNAME in $(echo "doc('./data/ganzeUrkunde/133_ganzeUrkunden_Seitenstetten.xml')/annotation/file/object/bbox/concat(../../@filename,codepoints-to-string(10))" | saxonb-xquery - | tail -c +39| dos2unix|sed 's/^ *//g'|sed 's/ /\\\ /g'); do cp /tmp/icarus/monasterium/pics/133/$FNAME ./data/images/ ; done

#Missing files:
#/tmp/icarus/monasterium/pics/133/K.._MOM-Bilddateien._~Seitenstettenjpgweb._~StAS_Eigene_Urkunden_1190_00_00.jpg
#/tmp/icarus/monasterium/pics/133/K.._MOM-Bilddateien._~Seitenstettenjpgweb._~StAS_Eigene_Urkunden_1190_00_00-v.jpg
```


Running YOLO l1out/l1in class experiments
```bash
BATCH_SZ=32
for REPLICATE in {1..4}; do
	for CL in $(cd ../cloned_sealds/l1out;ls ); do
		echo "DOING: ${REPLICATE} ${CL}" ;
		python train.py --img 640 --batch "$BATCH_SZ" --epochs 40 --data "../cloned_sealds/l1out/${CL}/seal_ds.yaml" --weights yolov5s.pt --name "l1out_${CL}_${REPLICATE}";
		python train.py --img 640 --batch "$BATCH_SZ" --epochs 40 --data "../cloned_sealds/l1in/${CL}/seal_ds.yaml" --weights yolov5s.pt --name "l1in_${CL}_${REPLICATE}";
	done;
done;
```




