# Beam Search Module

from modules import *
from dataset import *
from tqdm import tqdm
import numpy as n
import argparse
from pytorch3d.io.obj_io import save_obj as save_mesh_obj

k = 5 # Beam size

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('testing translation model')
    argparser.add_argument('--gpu', default=-1, help='gpu id')
    argparser.add_argument('--N', default=6, type=int, help='num of layers')
    argparser.add_argument('--dataset', default='face', help='dataset')
    argparser.add_argument('--batch', default=64, type=int, help='batch size')
    argparser.add_argument('--ckpt-dir', default='.', type=str, help='checkpoint path')
    argparser.add_argument('--epoch', default=1, help='epoch number')
    args = argparser.parse_args()
    args_filter = ['batch', 'gpu']
    exp_setting = '-'.join('{}'.format(v) for k, v in vars(args).items() if k not in args_filter)
    device = 'cpu' if args.gpu == -1 else 'cuda:{}'.format(args.gpu)

    dataset = get_dataset('face')
    dim_model = 256

    graph_pool = FaceGraphPool()
    model = make_face_model(N=args.N, dim_model=dim_model)
    ckpt_path = os.path.join(args.ckpt_dir, 'ckpt.'+str(args.epoch)+'.pt')
    with open(ckpt_path, 'rb') as f:
        model.load_state_dict(th.load(f, map_location=lambda storage, loc: storage))
    model = model.to(device)
    model.eval()
    test_iter = dataset(graph_pool, mode='infer', batch_size=args.batch, device=device, k=k)
    for i, g in enumerate(test_iter):
        with th.no_grad():
            src, output = model.infer(g, dataset.MAX_FACE_LENGTH*3-2, dataset.STOP_FACE_VERT_IDX, k, alpha=0.6)
            num_verts = src.shape[0] // k
            vertex = src[:num_verts, :][2:, :].float() / 31 - 0.5
            faces = output[0]
            eos = faces.index(1)
            faces = th.tensor(np.array(faces[1:eos]).reshape([(eos-1)//3, 3])) - 2
            print (faces)
            res_file = os.path.join(args.ckpt_dir, 'example_res.obj')
            save_mesh_obj(res_file, vertex, faces)
            exit(0)









