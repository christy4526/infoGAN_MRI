from __future__ import absolute_import, division, print_function
import argparse
import os

def argument_report(arg, end='\n'):
    d = arg.__dict__
    keys = d.keys()
    report = '{:15}    {}'.format('running_fold', d['running_fold'])+end
    report += '{:15}    {}'.format('memo', d['memo'])+end
    for k in sorted(keys):
        if k == 'running_fold' or k == 'memo':
            continue
        report += '{:15}    {}'.format(k, d[k])+end
    return report


def _base_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('model', type=str)
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--save_dir', type=str, default='results')
    parser.add_argument('--checkpoint_root', type=str, default='checkpoint')
    parser.add_argument('--devices', type=int, nargs='+', default=(0,1,2,3))
    parser.add_argument('--labels', type=str, nargs='+', default=('AD','CN'))
    parser.add_argument('--fold', type=int, default=5)
    parser.add_argument('--running_fold', type=int, default=0)
    parser.add_argument('--memo', type=str, default='')
    parser.add_argument('--vis_env', type=str, default='infoGAN')
    parser.add_argument('--vis_port', type=int, default=10002)
    parser.add_argument('--subject_ids_path', type=str,
                        default=os.path.join('data', 'subject_ids.pkl'))
    parser.add_argument('--diagnosis_path', type=str,
                        default=os.path.join('data', 'diagnosis.pkl'))
    parser.add_argument('--load_pkl', type=str, default ='false')
    parser.add_argument('--gm', type=str, default ='false')

    return parser


def train_args():
    parser = argparse.ArgumentParser(parents=[_base_parser()])
    parser.add_argument('--num_epoch', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--l2_decay', type=float, default=0.0005)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lrG', type=float, default=0.0002)
    parser.add_argument('--lrD', type=float, default=0.0002)
    parser.add_argument('--lr_gamma', type=float, default=0.999)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--z', type=int, default=128)
    parser.add_argument('--d_code', type=int, default=2)
    parser.add_argument('--c_code', type=int, default=2)
    parser.add_argument('--axis', type=int, default=0)
    parser.add_argument('--isize', type=int, default=79)
    parser.add_argument('--SUPERVISED', type=str, default='True')
    parser.add_argument('--lr_adam', type=float, default=1e-4)
    parser.add_argument('--std', type=float, default=0.02, help='for weight')

    args = parser.parse_args()
    return args
