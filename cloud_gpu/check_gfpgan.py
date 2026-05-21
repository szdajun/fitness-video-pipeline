import sys, types
import torchvision.transforms.functional as _F
_t = types.ModuleType('torchvision.transforms.functional_tensor')
_t.rgb_to_grayscale = _F.rgb_to_grayscale
sys.modules['torchvision.transforms.functional_tensor'] = _t

import gfpgan, os, inspect

pkg_dir = os.path.dirname(gfpgan.__file__)
print('GFPGAN 路径:', pkg_dir)

# 搜索 ROOT_DIR
for root, dirs, files in os.walk(pkg_dir):
    for f in files:
        if f.endswith('.py'):
            path = os.path.join(root, f)
            with open(path) as fh:
                for line in fh:
                    if 'ROOT_DIR' in line and '=' in line:
                        print(f'{os.path.relpath(path, pkg_dir)}: {line.strip()}')

# 查看 GFPGANer 调用的完整上下文
from gfpgan import GFPGANer
src = inspect.getsource(GFPGANer.__init__)
# 找 ROOT_DIR 使用
for line in src.split('\n'):
    if 'ROOT_DIR' in line or 'model_path' in line or 'load_file' in line:
        print('  ', line.strip())
