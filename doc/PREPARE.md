
### Mounting Images 

```bash
REMOTE='grazgpu' # atzenhofer@143.50.30.63
mkdir -p ./data/icarus_mirror/
sshfs grazgpu:/data/anguelos/monasterium/ ./data/icarus_mirror/  -o ro
```
After mounting, you should only care about

### Python requirements
```bash
pip install --user tqdm lxml fargv
```

### UNIX requirements
```bash
pip install --user tqdm lxml fargv
```