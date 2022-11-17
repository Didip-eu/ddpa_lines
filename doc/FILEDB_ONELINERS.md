# (WSL) Pipe all charters' atom_ids into into clipboard
```bash
awk '{print $0}' */*/*/atom_id.txt | clip.exe
```

# Get A list of all charters with no (downloaded) images

```bash
ls $(echo */*/*/image_urls.json) -l|awk '{ print $9 " "  $5 }'|grep -e ' 2$'
```


```bash
NOIMAGE_DIRS=$(ls $(echo */*/*/image_urls.json) -l|awk '{ print $9 " "  $5 }'|grep -e ' 2$'|xargs dirname|grep -e '...*')
for DIRNAME in $NOIMAGE_DIRS;
do
echo "$DIRNAME/failed.txt"
done;
#scroll up to see the f*.txt vs failed.txt
```

