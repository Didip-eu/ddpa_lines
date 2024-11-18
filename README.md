# Line segmentation

+ wrapper around Kraken segmentation API
+ Line viewer
+ libraries for evaluating and visualizing segmentations (provided as PageXML or JSON data)


## TODO


## How to use


### Line detection

```bash
python3 ./bin/ddp_line_detect.py [ -<option> ... ]
```

where optional flags are one or more of the following:

```bash
-appname=<class 'str'>  Default 'lines' .
-model_path=<class 'str'>  Default '/tmp/blla.mlmodel' .
-img_paths=<class 'set'>  Default set() .
-charter_dirs=<class 'set'>  Default {'./'} .
-mask_classes=<class 'set'> Names of the seals-app regions on which lines are to be detected. Eg. '[Wr:OldText']. If empty (default), detection is run on the entire page. Default {'Wr:OldText'} .
-region_segmentation_suffix=<class 'str'> Regions are given by segmentation file that is <img name stem>.<suffix>. Default '.seals.pred.json' .
-preview=<class 'bool'>  Default False .
-preview_delay=<class 'int'>  Default 0 .
-dry_run=<class 'bool'>  Default False .
-just_show=<class 'bool'>  Default False .
-line_type=<class 'tuple'> Line segmentation type: polygon = Kraken (CNN-inferred) baselines + polygons; legacy_bbox: legacy Kraken segmentation) Default ('polygon', 'legacy_bbox') .
-output_format=<class 'tuple'> Segmentation output: xml=<Page XML>, json=<JSON file>, tensor=<a (4,H,W) label map where each pixel can store up to 4 labels (for overlapping polygons) Default ('xml', 'json', 'pt') .
-help=<class 'bool'> Print help and exit. Default False .
-bash_autocomplete=<class 'bool'> Print a set of bash commands that enable autocomplete for current program. Default False .
-h=<class 'bool'> Print help and exit Default False .
-v=<class 'int'> Set verbosity level. Default 1 .
```


### Line viewer



```bash
python3 ./bin/ddp_line_detect.py [ -<option> ... ]
```
where optional flags are one or more of the following:

```bash
-appname=<class 'str'>  Default 'lines' .
-img_paths=<class 'set'>  Default set() .
-charter_dirs=<class 'set'>  Default {'./'} .
-mask_classes=<class 'set'> Names of the seals-app regions on which lines are to be detected. Eg. '[Wr:OldText']. If empty (default), detection is run on the entire page. Default {'Wr:OldText'} .
-region_segmentation_suffix=<class 'str'> Regions are given by segmentation file that is <img name stem>.<suffix>. Default '.seals.pred.json' .
-colors=<class 'int'> Number of colors in the palette. Default 2 .
-style=<class 'tuple'> Display style: 'map' for pixel map (slow); 'outline' for polygon lines. Default ('outline', 'map') .
-help=<class 'bool'> Print help and exit. Default False .
-bash_autocomplete=<class 'bool'> Print a set of bash commands that enable autocomplete for current program. Default False .
-h=<class 'bool'> Print help and exit Default False .
-v=<class 'int'> Set verbosity level. Default 1 .

```



