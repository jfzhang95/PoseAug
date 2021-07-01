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
    parser.add_argument('--snapshot', default=2, type=int, help='save gan rlt for every #snapshot epochs')
    parser.add_argument('--note', default='debug', type=str, help='additional name on checkpoint directory')

    # Evaluate choice
    parser.add_argument('--evaluate', default='', type=str, metavar='FILENAME',
                        help='checkpoint to evaluate (file name)')  # not in used here.
    parser.add_argument('--resume', default='', type=str, metavar='FILENAME',
                        help='checkpoint to resume (file name)')

    # Model arguments
    parser.add_argument('--posenet_name', default='videopose', type=str, help='posenet: gcn/stgcn/videopose/mlp')
    parser.add_argument('--stages', default=4, type=int, metavar='N', help='stages of baseline model')
    parser.add_argument('--dropout', default=0.25, type=float, help='dropout rate')

    # Training detail
    parser.add_argument('--batch_size', default=1024, type=int, metavar='N',
                        help='batch size in terms of predicted frames')
    parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of training epochs')
    parser.add_argument('--decay_epoch', default=0, type=int, metavar='N', help='number of decay epochs')

    # Learning rate
    parser.add_argument('--lr_g', default=1.0e-4, type=float, metavar='LR', help='initial learning rate for augmentor/generator')
    parser.add_argument('--lr_d', default=1.0e-4, type=float, metavar='LR', help='initial learning rate for discriminator')
    parser.add_argument('--lr_p', default=1.0e-4, type=float, metavar='LR', help='initial learning rate for posenet')
    parser.add_argument('--no_max', dest='max_norm', action='store_false', help='if use max_norm clip on grad')
    parser.set_defaults(max_norm=True)

    # Experimental setting
    parser.add_argument('--random_seed', type=int, default=0)  # change this if GAN collapse
    parser.add_argument('--downsample', default=1, type=int, metavar='FACTOR', help='downsample frame rate by factor')
    parser.add_argument('--pretrain', default=True, type=lambda x: (str(x).lower() == 'true'), help='pretrain model')
    parser.add_argument('--s1only', default=False, type=lambda x: (str(x).lower() == 'true'), help='train S1 only')
    parser.add_argument('--num_workers', default=2, type=int, metavar='N', help='num of workers for data loading')

    # Training PoseAug detail
    parser.add_argument('--warmup', default=2, type=int, help='train gan only at the beginning')
    parser.add_argument('--df', default=2, type=int, help='update discriminator frequency')

    parser.add_argument('--hardratio_std_ba', default=2, type=float, help='hard ratio accepted range')
    parser.add_argument('--hardratio_ba_s', default=3, type=float, help='hard ratio at the starting point')
    parser.add_argument('--hardratio_ba', default=5, type=float, help='hard ratio at the end point')
    parser.add_argument('--hardratio_std_rt', default=15, type=float, help='hard ratio accepted range')
    parser.add_argument('--hardratio_rt_s', default=17, type=float, help='hard ratio at the starting point')
    parser.add_argument('--hardratio_rt', default=17, type=float, help='hard ratio at the end point')

    parser.add_argument('--ba_range_m', default=20.5e-2, type=float, help='bone angle modification range.')
    parser.add_argument('--ba_range_w', default=16.5e-2, type=float, help='bone angle modification range.')

    parser.add_argument('--blr_tanhlimit', default=2e-1, type=float, help='bone length change limit.')
    parser.add_argument('--blr_limit', default=1e-1, type=float, help='bone length change limit.')

    parser.add_argument('--gloss_factord2d', default=1, type=float, help='factor for d2d loss.')
    parser.add_argument('--gloss_factord3d', default=6, type=float, help='factor for d3d loss.')
    parser.add_argument('--gloss_factordiff', default=3, type=float, help='factor for range difference loss.')
    parser.add_argument('--gloss_factorblr', default=1, type=float, help='factor for blr loss.')

    parser.add_argument('--gloss_factorfeedback_ba', default=1e-1, type=float, help='factor for feedback loss from ba.')
    parser.add_argument('--gloss_factordiv_ba', default=0., type=float, help='not in use.')
    parser.add_argument('--gloss_factorfeedback_rt', default=1e-2, type=float, help='factor for feedback loss from rt.')
    parser.add_argument('--gloss_factordiv_rt', default=0., type=float, help='not in use.')

    args = parser.parse_args()

    # for debug
    if args.note == 'debug':
        args.pretrain = False

    # Check invalid configuration
    if args.resume and args.evaluate:
        print('Invalid flags: --resume and --evaluate cannot be set at the same time')
        exit()

    return args

