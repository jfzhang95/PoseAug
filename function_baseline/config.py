import argparse



def get_parse_args():
    parser = argparse.ArgumentParser(description='PyTorch training script')

    # General arguments
    parser.add_argument('--dataset', default='h36m', type=str, metavar='NAME', help='target dataset')
    parser.add_argument('--keypoints', default='gt', type=str, metavar='NAME', help='2D detections to use, \
    gt/hr/cpn_ft_h36m_dbb/detectron_ft_h36m')
    parser.add_argument('--actions', default='*', type=str, metavar='LIST',
                        help='actions to train/test on, separated by comma, or * for all')
    parser.add_argument('--checkpoint', default='checkpoint/debug', type=str, metavar='PATH',
                        help='checkpoint directory')
    parser.add_argument('--snapshot', default=25, type=int, help='save models_baseline for every (default: 20)')
    parser.add_argument('--note', default='debug', type=str, help='additional name on checkpoint directory')

    # Evaluate choice
    parser.add_argument('--evaluate', default='', type=str, metavar='FILENAME',
                        help='checkpoint to evaluate (file name)')
    parser.add_argument('--action-wise', default=True, type=lambda x: (str(x).lower() == 'true'), help='train s1only')

    # Model arguments
    parser.add_argument('--posenet_name', default='videopose', type=str, help='posenet: gcn/stgcn/videopose/mlp')
    parser.add_argument('--stages', default=4, type=int, metavar='N', help='stages of baseline model')
    parser.add_argument('--dropout', default=0.25, type=float, help='dropout rate')

    # Training detail
    parser.add_argument('--batch_size', default=1024, type=int, metavar='N',
                        help='batch size in terms of predicted frames')
    parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of training epochs')

    # Learning rate
    parser.add_argument('--lr', default=1.0e-3, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--lr_decay', type=int, default=100000, help='num of steps of learning rate decay')
    parser.add_argument('--lr_gamma', type=float, default=0.96, help='gamma of learning rate decay')
    parser.add_argument('--no_max', dest='max_norm', action='store_false', help='if use max_norm clip on grad')
    parser.set_defaults(max_norm=True)

    # Experimental setting
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--downsample', default=1, type=int, metavar='FACTOR', help='downsample frame rate by factor')
    parser.add_argument('--pretrain', default=False, type=lambda x: (str(x).lower() == 'true'), help='used in poseaug')
    parser.add_argument('--s1only', default=False, type=lambda x: (str(x).lower() == 'true'), help='train S1 only')
    parser.add_argument('--num_workers', default=2, type=int, metavar='N', help='num of workers for data loading')

    args = parser.parse_args()

    return args

