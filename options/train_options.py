

"""
    Options to be used while training

    Version 1 2019-07-12 Abhinav Kumar More options added
    Version 1 2019-06-25 Abhinav Kumar More options added
    Version 0 2017-05-xx Xi Peng
"""

from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--layer_num', type=int, default=8,
                                 help='number of coupled U-Nets')
        self.parser.add_argument('--order', type=int, default=1,
                                 help='order-K coupling')
        self.parser.add_argument('--class_num', type=int, default=68, #16 originally
                                 help='number of classes in the prediction')
        self.parser.add_argument('--loss_num', type=int, default=2,
                                 help='number of losses in the CU-Net')
        self.parser.add_argument('--optimizer', type=str, default="rmsprop",
                                help='optimizer to be used (rmsprop | adam) (default: rmsprop)')
        self.parser.add_argument('--lr', type=float, default=1e-4,
                                 help='initial learning rate (default: 1e-4)')
        self.parser.add_argument('--bs', type=int, default=24,
                                 help='mini-batch size')
        self.parser.add_argument('--load_checkpoint', type=bool, default=False,
                                 help='use checkpoint model')
        self.parser.add_argument('--adjust_lr', type=bool, default=False,
                                 help='adjust learning rate')
        self.parser.add_argument('--resume_prefix', type=str, default='',
                                 help='checkpoint name for resuming')
        self.parser.add_argument('--nEpochs', type=int, default=50,
                                 help='number of total training epochs to run (default:30)')
        self.parser.add_argument('--best_pckh', type=float, default=0.,
                                 help='best result until now')
        self.parser.add_argument('--print_freq', type=int, default=50,
                                 help='print log every n iterations')
        self.parser.add_argument('--display_freq', type=int, default=10,
                                 help='display figures every n iterations')
        self.parser.add_argument('--bits_w', type=int, default=1,
                                help='bits of weight')
        self.parser.add_argument('--bits_i', type=int, default=8,
                                help='bits of input')
        self.parser.add_argument('--bits_g', type=int, default=8,
                                help='bits of gradient')
        self.parser.add_argument('--freeze', action='store_true',
                                help='freeze the basenet')
        self.parser.add_argument('--hg_wt', type=str, default="1,1,1,1,1,1,1,1",
                                help='weights of hourglasses separated by comma. These weights are not normalized in the code. (default=\"1,1,1,1,1,1,1,1\")')
        self.parser.add_argument('--pp' , type=str, default="",
                                help='post processing of heatmaps (default:none)')
        self.parser.add_argument('--smax', action='store_true',
                                help='use softmax in training')
        self.parser.add_argument('--tau', type=float, default=0.02,
                                help='scaling parameter for softmax (default: 0.02)')
        self.parser.add_argument('-w', '--saved_wt_file', type=str, default="face-layer-num-8-order-1-model-best.pth.tar",
                                help='saved model pth.tar file from which weights are loaded')
        self.parser.add_argument('--json_path' , type=str, default=".",
                                help='json path (default=\".\")')
        self.parser.add_argument('--train_json', type=str, default="dataset/normal_train.json",
                                help='training json to be used. (default=\"dataset/normal_train.json\")')
        self.parser.add_argument('--val_json', type=str, default="dataset/normal_val.json",
                                help='training json to be used. (default=\"dataset/normal_val.json\")')
        self.parser.add_argument('--measure', type=str, default="nothing",
                                help='regularization measure to be used. (default=\"nothing\")')
        self.parser.add_argument('--wt_mse', type=float, default=0.0,
                                help='weight of the MSE term (default=0.0)')
        self.parser.add_argument('--wt_gau', type=float, default=1.0,
                                help='weight of the Gaussian Log Likelihood (default=1.0)')
        self.parser.add_argument('--wt_gauss_regln', type=float, default=0.0,
                                help='weight of the Gaussian Log Likelihood (default=0.0)')
        self.parser.add_argument('--is_covariance', action='store_true', 
                                help='use covariance from the heatmaps')
        self.parser.add_argument('--get_mean_from_mlp', action='store_true',
                                help="uses mean from MLP instead of the heatmaps") 
        self.parser.add_argument('--bulat_aug'    , action='store_true', 
                                help='use bulat augmentation in training')
        self.parser.add_argument('--stn', action='store_true', 
                                help='use spatial transformer networks after heatmaps')
        self.parser.add_argument('--lr_policy', default='1',
                                help="Choose one of the learning rate policies to run.")
        self.parser.add_argument('--slurm', action='store_true',
                                help="use slurm for resources. So, ignore gpu-id")
        self.parser.add_argument('--flipped', action='store_true',
                                help="use mean of the flipped image for evaluation for pose-estimation")
        self.parser.add_argument('--mlp_tot_layers', type=int, default=1,
                                help="number of total layers in the MLP/Cholesky calculator = number of hidden layers + 1 (default= 1)")
        self.parser.add_argument('--mlp_hidden_units', type=int, default=4096,
                                help="number of hidden units in each hidden layer of MLP/Cholesky calculator (default= 4096)")
        self.parser.add_argument('--laplacian', action='store_true',
                                help="use laplacian likelihood instead of Gaussian while training")
        self.parser.add_argument('--laplacian_form', default="simplified",
                                help="type/form of laplacian to be used in training - asymmetric, symmetric, simplified (default= simplified)")
        self.parser.add_argument('--use_visibility', action='store_true',
                                help="use probability of visible also while training")
        self.parser.add_argument('--use_heatmaps', action='store_true',
                                help="use proxy ground truth heatmaps in training")                         
        self.parser.add_argument('--save_image_heatmaps', action='store_true',
                                help="save_images as well")
        self.parser.add_argument('-s', '--split', type= int, default= 1, 
                                help= 'split to use (default:1)')
        # self.parser.add_argument('--momentum', type=float, default=0.90,
        #             help='momentum term of sgd')
        # self.parser.add_argument('--weight_decay', type=float, default=1e-4,
        #             help='weight decay term of sgd')
        # self.parser.add_argument('--beta1', type=float, default=0.5,
        #             help='momentum term of adam')
