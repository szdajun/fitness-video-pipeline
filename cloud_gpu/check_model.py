"""检查 GFPGAN 模型下载位置"""
import sys, types
import torchvision.transforms.functional as _F
_tensor_mod = types.ModuleType('torchvision.transforms.functional_tensor')
_tensor_mod.rgb_to_grayscale = _F.rgb_to_grayscale
sys.modules['torchvision.transforms.functional_tensor'] = _tensor_mod

import gfpgan
import os

pkg_dir = os.path.dirname(gfpgan.__file__)
print('Package dir:', pkg_dir)

# Check init for ROOT_DIR
init_path = os.path.join(pkg_dir, '__init__.py')
with open(init_path) as f:
    for line in f:
        if 'ROOT_DIR' in line or 'root_dir' in line or 'WEIGHTS' in line:
            print('Init:', line.strip())

# Check if there's a separate module for paths
import gfpgan.utils as utils
print('Utils file:', utils.__file__)

# Check for version
print('GFPGAN version:', getattr(gfpgan, '__version__', 'unknown'))

# Check for GFPGANer module
gfpganer_path = os.path.join(pkg_dir, 'models')
if os.path.isdir(gfpganer_path):
    for fname in os.listdir(gfpganer_path):
        print('Model file:', fname)

# Look for any .pth in the package
for root, dirs, files in os.walk(pkg_dir):
    for f in files:
        if f.endswith('.pth'):
            print(f'Found .pth: {os.path.join(root, f)}')

# Check basicsr download URL
import basicsr
bsr_dir = os.path.dirname(basicsr.__file__)
print('\nBasicsr dir:', bsr_dir)
try:
    from basicsr.utils.download import load_file_from_url
    print('download.py:', load_file_from_url.__module__)
except Exception as e:
    print(f'load_file_from_url error: {e}')

# Look at download.py source if available
for root, dirs, files in os.walk(bsr_dir):
    for f in files:
        if f == 'download.py':
            path = os.path.join(root, f)
            with open(path) as fh:
                content = fh.read()
                for line in content.split('\n'):
                    if 'url' in line.lower() and ('https' in line or 'pth' in line or 'model' in line):
                        print(f'  {line.strip()}')
            break
