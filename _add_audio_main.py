import subprocess, shutil, os
from pathlib import Path

FFMPEG = 'C:/Users/18091/ffmpeg/ffmpeg.exe'

for name, out_dir in [('小飞侠3', 'output/小飞侠3_2026-04-24'),
                       ('小红豆2', 'output/小红豆2_2026-04-24'),
                       ('李刚3', 'output/李刚3_2026-04-24')]:
    main = Path(out_dir) / f'{name}_h_main.mp4'
    tmp = Path('F:/wkspace/temp') / f'_main_{name}.mp4'
    r = subprocess.run([FFMPEG, '-y', '-i', str(main),
                       '-f', 'lavfi', '-i', 'anullsrc=r=44100:cl=stereo',
                       '-c:v', 'copy', '-c:a', 'aac', '-shortest',
                       str(tmp)],
                      capture_output=True, text=True, encoding='utf-8', errors='replace')
    if r.returncode == 0:
        shutil.copy2(tmp, main)
        tmp.unlink()
        print(f'{name}: OK')
    else:
        print(f'{name}: FAIL')