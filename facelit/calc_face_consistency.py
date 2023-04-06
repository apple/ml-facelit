import click
import glob
import os 
from deepface import DeepFace
from tqdm import tqdm
import numpy as np 

@click.command()
@click.option('--img-dir', 'img_dir', help='image dir', required=True)

def calc_face_consistency_metrics(img_dir):
    view1 = sorted(glob.glob(os.path.join(img_dir, '*view1.png')))
    view2 = sorted(glob.glob(os.path.join(img_dir, '*view2.png')))
    total = []
    for v1, v2 in zip(view1, view2):
        assert v1[:-8] == v2[:-8]
        try:
            result = DeepFace.verify(v1, v2,
                model_name="ArcFace", 
                distance_metric="cosine", 
                enforce_detection=False,
                detector_backend="mtcnn",
                prog_bar=False)
        except:
            continue
        total.append(1 - result['distance'])
        if len(total) == 1024:
            break

    print(f"Cosine Similarity: {np.mean(total)}")
    
    with open('eval_metrics.log', "a") as f:
        print(f"Img dir {img_dir}, ID Cosine Similarity: {np.mean(total)}", file=f)

if __name__ == '__main__':
    calc_face_consistency_metrics()
