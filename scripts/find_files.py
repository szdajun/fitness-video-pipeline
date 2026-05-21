import os, glob

# 找到最新的 h2v_warped 文件
files = glob.glob('output/*h2v_warped.mp4')
for f in sorted(files):
    print(repr(f), os.path.getsize(f), os.path.getmtime(f))

print()

# 找到所有h2v文件
files2 = glob.glob('output/*h2v.mp4')
for f in sorted(files2):
    print(repr(f), os.path.getsize(f), os.path.getmtime(f))