"""查找模型下载 URL"""
import sys, types
import torchvision.transforms.functional as _F
_t = types.ModuleType('torchvision.transforms.functional_tensor')
_t.rgb_to_grayscale = _F.rgb_to_grayscale
sys.modules['torchvision.transforms.functional_tensor'] = _t

import gfpgan, os

# 读 utils.py 了解 model_path 默认下载地址
utils_py = os.path.join(os.path.dirname(gfpgan.__file__), 'utils.py')
with open(utils_py) as f:
    content = f.read()
    # 找出 load_file_from_url 导入
    for line in content.split('\n'):
        if 'load_file_from_url' in line or 'url' in line or 'URL' in line or 'release' in line or 'github' in line.lower():
            print(line)

# 看 facexlib 下载地址
import facexlib
fx_dir = os.path.dirname(facexlib.__file__)
print('\n--- facexlib ---')
for root, dirs, files in os.walk(fx_dir):
    for f in files:
        if f.endswith('.py') and 'detection' in root:
            path = os.path.join(root, f)
            with open(path) as fh:
                for line in fh:
                    if 'url' in line.lower() or 'release' in line.lower() or 'pth' in line.lower():
                        print(f'{os.path.relpath(path, fx_dir)}: {line.strip()}')

# basicsr load_file_from_url
import basicsr
bsr_dir = os.path.dirname(basicsr.__file__)
print('\n--- basicsr download ---')
for root, dirs, files in os.walk(bsr_dir):
    for f in files:
        if f == 'download.py':
            path = os.path.join(root, f)
            with open(path) as fh:
                print(fh.read())
            break
