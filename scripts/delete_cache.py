import os
p = 'F:/wkspace/fitness-video-pipeline/output/秀秀_20260222_203724_original_keypoints.json'
if os.path.exists(p):
    os.remove(p)
    print('deleted:', p)
else:
    print('not found:', p)