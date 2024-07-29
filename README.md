# Line segmentation

+ wrapper around Kraken segmentation API
+ libraries for evaluating and visualizing segmentations (provided as PageXML or JSON data)

## Examples::

	export PYTHONPATH=.

	# PageXML output
	python3 ./bin/ddp_line_detect.py -img_paths /home/nicolas/tmp/data/1000CV/SK-SNA/f5dc4a3628ccd5307b8e97f02d9ff12a/89ce0542679f64d462a73f7e468ae812/*img.jpg -output_format xml

    # Pickled polygon map from existing PageXML
	python3 ./bin/ddp_line_detect.py -img_paths /home/nicolas/tmp/data/1000CV/SK-SNA/f5dc4a3628ccd5307b8e97f02d9ff12a/89ce0542679f64d462a73f7e468ae812/*img.jpg -mapify

	# Output is a polygon tensor, allowing for easy visualization
	python3 ./bin/ddp_line_detect.py -img_paths /home/nicolas/tmp/data/1000CV/SK-SNA/f5dc4a3628ccd5307b8e97f02d9ff12a/89ce0542679f64d462a73f7e468ae812/*img.jpg -output_format pt
	python3 ./bin/ddp_line_detect.py -img_paths /home/nicolas/tmp/data/1000CV/SK-SNA/f5dc4a3628ccd5307b8e97f02d9ff12a/89ce0542679f64d462a73f7e468ae812/*img.jpg -just_show

## Options::

	-appname=<class 'str'>  Default 'lines'.
	-model_path=<class 'pathlib.PosixPath'>  Default PosixPath('/home/nicolas/tmp/models/segmentation/blla.mlmodel').
	-img_paths=<class 'set'>  Default {PosixPath('/home/nicolas/tmp/data/1000CV/AT-AES/d3a416ef7813f88859c305fb83b20b5b/207cd526e08396b4255b12fa19e8e4f8/4844ee9f686008891a44821c6133694d.img.jpg')}.
	-preview=<class 'bool'>  Default False.
	-preview_delay=<class 'int'>  Default 0.
	-dry_run=<class 'bool'>  Default False.
	-just_show=<class 'bool'>  Default False.
	-mapify=<class 'bool'>  Default False.
    -line_type=<class 'tuple'> Line segmentation type: polygon = Kraken (CNN-inferred) baselines + polygons; bbox = bounding boxes, derived from the former; legacy_bbox: legacy Kraken segmentation) Default ('polygon').
	-output_format=<class 'tuple'> Segmentation output: xml=<Page XML>, json=<JSON file>, tensor=<a (4,H,W) label map where each pixel can store up to 4 labels (for overlapping polygons) Default ('xml').
	-help=<class 'bool'> Print help and exit. Default False.
	-bash_autocomplete=<class 'bool'> Print a set of bash commands that enable autocomplete for current program. Default False.
	-h=<class 'bool'> Print help and exit Default False.
	-v=<class 'int'> Set verbosity level. Default 1.

