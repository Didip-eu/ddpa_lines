# Line segmentation

+ wrapper around Kraken segmentation API
+ libraries for evaluating and visualizing segmentations (provided as PageXML or JSON data)

## Examples::

	export PYTHONPATH

	# PageXML output
	./bin/ddp_line_detect.py -img_paths /home/nicolas/tmp/data/1000CV/SK-SNA/f5dc4a3628ccd5307b8e97f02d9ff12a/89ce0542679f64d462a73f7e468ae812/*img.jpg -output_format xml

	# Output is a polygon tensor, allowing for easy visualization
	./bin/ddp_line_detect.py -img_paths /home/nicolas/tmp/data/1000CV/SK-SNA/f5dc4a3628ccd5307b8e97f02d9ff12a/89ce0542679f64d462a73f7e468ae812/*img.jpg -output_format pt
	./bin/ddp_line_detect.py -img_paths /home/nicolas/tmp/data/1000CV/SK-SNA/f5dc4a3628ccd5307b8e97f02d9ff12a/89ce0542679f64d462a73f7e468ae812/*img.jpg -just_show
