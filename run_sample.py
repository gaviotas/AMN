import argparse
import os

from misc import pyutils

if __name__ == '__main__':

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser()

    # Environment
    parser.add_argument("--num_workers", default=os.cpu_count()//2, type=int)
    parser.add_argument("--voc12_root", required=True, type=str,
                        help="Path to VOC 2012 Devkit, must contain ./JPEGImages as subdirectory.")

    # Dataset
    parser.add_argument("--train_list", default="voc12/train_aug.txt", type=str)
    parser.add_argument("--val_list", default="voc12/val.txt", type=str)
    parser.add_argument("--infer_list", default="voc12/train.txt", type=str,
                        help="voc12/train_aug.txt to train a fully supervised model, "
                             "voc12/train.txt or voc12/val.txt to quickly check the quality of the labels.")
    parser.add_argument("--chainer_eval_set", default="train", type=str)

    # Class Activation Map
    parser.add_argument("--cam_network", default="net.resnet50_cam", type=str)
    parser.add_argument("--cam_crop_size", default=512, type=int)
    parser.add_argument("--cam_batch_size", default=16, type=int)
    parser.add_argument("--cam_num_epoches", default=5, type=int)
    parser.add_argument("--cam_learning_rate", default=0.1, type=float)
    parser.add_argument("--cam_weight_decay", default=1e-4, type=float)
    parser.add_argument("--cam_eval_thres", default=0.25, type=float)
    parser.add_argument("--cam_scales", default=(1.0, 0.5, 1.5, 2.0),
                        help="Multi-scale inferences")

    # AMN
    parser.add_argument("--amn_network", default="net.resnet50_amn", type=str)
    parser.add_argument("--amn_crop_size", default=512, type=int)
    parser.add_argument("--amn_batch_size", default=16, type=int)
    parser.add_argument("--amn_num_epoches", default=5, type=int)
    parser.add_argument("--eps", default=0.4, type=float)


    # Mining Inter-pixel Relations
    parser.add_argument("--conf_fg_thres", default=0.45, type=float)
    parser.add_argument("--conf_bg_thres", default=0.15, type=float)

    # Inter-pixel Relation Network (IRNet)
    parser.add_argument("--irn_network", default="net.resnet50_irn", type=str)
    parser.add_argument("--irn_crop_size", default=512, type=int)
    parser.add_argument("--irn_batch_size", default=32, type=int)
    parser.add_argument("--irn_num_epoches", default=3, type=int)
    parser.add_argument("--irn_learning_rate", default=0.1, type=float)
    parser.add_argument("--irn_weight_decay", default=1e-4, type=float)

    # Random Walk Params
    parser.add_argument("--beta", default=10)
    parser.add_argument("--exp_times", default=8,
                        help="Hyper-parameter that controls the number of random walk iterations,"
                             "The random walk is performed 2^{exp_times}.")
    parser.add_argument("--ins_seg_bg_thres", default=0.25)
    parser.add_argument("--sem_seg_bg_thres", default=0.25)

    # Output Path
    parser.add_argument("--log_name", default="sample_train_eval", type=str)
    parser.add_argument("--cam_weights_name", default="sess/res50_cam.pth", type=str)
    parser.add_argument("--amn_weights_name", default="sess/res50_amn.pth", type=str)
    parser.add_argument("--irn_weights_name", default="sess/res50_irn.pth", type=str)

    parser.add_argument("--cam_out_dir", default="result/cam", type=str)
    parser.add_argument("--ir_label_out_dir", default="result/ir_label", type=str)
    parser.add_argument("--amn_cam_out_dir", default="result/amn_cam", type=str)
    parser.add_argument("--amn_ir_label_out_dir", default="result/amn_ir_label", type=str)
    parser.add_argument("--sem_seg_out_dir", default="result/sem_seg", type=str)
    parser.add_argument("--ins_seg_out_dir", default="result/ins_seg", type=str)

    # Step
    parser.add_argument("--train_cam_pass", type=str2bool, default=False)
    parser.add_argument("--make_cam_pass", type=str2bool, default=False)
    parser.add_argument("--eval_cam_pass", type=str2bool, default=False)
    parser.add_argument("--cam_to_ir_label_pass", type=str2bool, default=False)

    parser.add_argument("--train_amn_pass", type=str2bool, default=False)
    parser.add_argument("--make_amn_cam_pass", type=str2bool, default=False)
    parser.add_argument("--eval_amn_cam_pass", type=str2bool, default=False)
    parser.add_argument("--amn_cam_to_ir_label_pass", type=str2bool, default=False)

    parser.add_argument("--train_irn_pass", type=str2bool, default=False)
    parser.add_argument("--make_ins_seg_pass", type=str2bool, default=False)
    parser.add_argument("--eval_ins_seg_pass", type=str2bool, default=False)
    parser.add_argument("--make_sem_seg_pass", type=str2bool, default=False)
    parser.add_argument("--eval_sem_seg_pass", type=str2bool, default=False)

    args = parser.parse_args()

    os.makedirs("sess", exist_ok=True)
    os.makedirs(args.cam_out_dir, exist_ok=True)
    os.makedirs(args.ir_label_out_dir, exist_ok=True)

    os.makedirs(args.amn_cam_out_dir, exist_ok=True)
    os.makedirs(args.amn_ir_label_out_dir, exist_ok=True)

    os.makedirs(args.sem_seg_out_dir, exist_ok=True)
    os.makedirs(args.ins_seg_out_dir, exist_ok=True)

    pyutils.Logger(args.log_name + '.log')
    print(vars(args))

    if args.train_cam_pass is True:
        import step.train_cam

        timer = pyutils.Timer('step.train_cam:')
        step.train_cam.run(args)

    if args.make_cam_pass is True:
        import step.make_cam

        timer = pyutils.Timer('step.make_cam:')
        step.make_cam.run(args)

    if args.eval_cam_pass is True:
        import step.eval_cam

        timer = pyutils.Timer('step.eval_cam:')
        step.eval_cam.run(args)

    if args.cam_to_ir_label_pass is True:
        import step.cam_to_ir_label

        timer = pyutils.Timer('step.cam_to_ir_label:')
        step.cam_to_ir_label.run(args)

    #### AMN training process ####

    if args.train_amn_pass is True:
        import step.train_amn

        timer = pyutils.Timer('step.train_cam:')
        step.train_amn.run(args)

    if args.make_amn_cam_pass is True:
        import step.make_amn_cam

        timer = pyutils.Timer('step.make_cam:')
        step.make_amn_cam.run(args)

    if args.eval_amn_cam_pass is True:
        import step.eval_amn_cam

        timer = pyutils.Timer('step.eval_cam:')
        step.eval_amn_cam.run(args)

    if args.amn_cam_to_ir_label_pass is True:
        import step.amn_cam_to_ir_label

        timer = pyutils.Timer('step.cam_to_ir_label:')
        step.amn_cam_to_ir_label.run(args)

    if args.train_irn_pass is True:
        import step.train_irn

        timer = pyutils.Timer('step.train_irn:')
        step.train_irn.run(args)

    if args.make_ins_seg_pass is True:
        import step.make_ins_seg_labels

        timer = pyutils.Timer('step.make_ins_seg_labels:')
        step.make_ins_seg_labels.run(args)

    if args.eval_ins_seg_pass is True:
        import step.eval_ins_seg

        timer = pyutils.Timer('step.eval_ins_seg:')
        step.eval_ins_seg.run(args)

    if args.make_sem_seg_pass is True:
        import step.make_sem_seg_labels

        timer = pyutils.Timer('step.make_sem_seg_labels:')
        step.make_sem_seg_labels.run(args)

    if args.eval_sem_seg_pass is True:
        import step.eval_sem_seg

        timer = pyutils.Timer('step.eval_sem_seg:')
        step.eval_sem_seg.run(args)

