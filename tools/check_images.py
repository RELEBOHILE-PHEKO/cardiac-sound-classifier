from PIL import Image
from pathlib import Path
imgs = ['class_distribution.png','training_history.png','confusion_matrix.png','roc_curve.png']
for name in imgs:
    p = Path('outputs')/name
    if not p.exists():
        print('MISSING', p)
        continue
    try:
        im = Image.open(p)
        im.verify()
        im2 = Image.open(p)
        print('OK', p, im2.format, im2.size, im2.mode)
    except Exception as e:
        print('ERR', p, repr(e))
