import sys, importlib
sys.path.insert(0, 'F:/wkspace/fitness-video-pipeline')
print("Python:", sys.executable)
try:
    m = importlib.import_module('stages.27_face_beautify2')
    print('27_face_beautify2 imported ok')
    print('FaceAnalysis:', m.FaceAnalysis)
except Exception as e:
    print('Import error:', e)
    import traceback
    traceback.print_exc()