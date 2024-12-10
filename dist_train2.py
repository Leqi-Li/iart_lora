import os
import sys
import argparse
from os import path as osp
from basicsr.models import build_model
from basicsr.utils.options import copy_opt_file, dict2str, parse_options
from recurrent_mix_precision_train2 import train_pipeline


def find_files_in_directory(directory):  
    """  
    遍历指定目录（不包括子目录）下的所有.yml文件，并返回它们的完整路径列表。  
  
    :param directory: 需要遍历的目录路径（字符串）  
    :return: 包含所有.txt文件路径的列表  
    """  
    yml_files = []  # 初始化一个空列表来存储.yml文件的路径  
  
    # 检查目录是否存在  
    if not os.path.isdir(directory):  
        raise FileNotFoundError(f"指定的目录 {directory} 不存在。")  
  
    # 遍历目录下的所有文件和文件夹  
    for filename in os.listdir(directory):  
        # 拼接完整的文件路径  
        filepath = os.path.join(directory, filename)  
        # 检查文件是否为.yml类型  
        if os.path.isfile(filepath) and filepath.endswith('.yml'):  
            yml_files.append(filepath)  
  
    return yml_files  
  
def setup_args(current_yml):
    # Setup your command line arguments manually here
    # current_yml_file = '/home/leqi/codes/IART-mainv2/' + current_yml
    current_yml_file = 'options/' + current_yml
    print('current_yml_file in setup_args:', current_yml_file)
    sys.argv = [
        'recurrent_mix_precision_train.py',  # Simulate script name
        '-opt', '/home/leqi/codes/IART-mainv2/options/IART_REDS_N6_300K.yml',       # Configuration file path
        # '-opt', current_yml_file,       # Configuration file path
        '--launcher', 'none',             # Launcher type
        '--auto_resume',                     # Include this flag if you want auto-resume
        # '--debug',                           # Include this flag to enable debug mode
        '--local_rank=0',                    # Local rank for distributed training
        # Add any additional flags or options as needed
    ]

def main(current_yml):
    setup_args(current_yml)

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    # # Set up PYTHONPATH as in the original bash script
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
    # current_pythonpath = os.environ.get('PYTHONPATH', '')
    # os.environ['PYTHONPATH'] = f"{parent_dir}:{current_pythonpath}"

    # # Assuming train_pipeline and osp are from the same module or correctly imported
    # from recurrent_mix_precision_train import train_pipeline
    # import os.path as osp

    # # Main entry logic
    # root_path = osp.abspath(osp.join(__file__, osp.pardir))
    # train_pipeline(root_path)
    
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    opt, args = parse_options(root_path, is_train=True)
    opt['root_path'] = root_path
    # opt['auto_resume'] = True
    
    #修正fix_flow: 5000为25, opt['train']['fix_flow']
    # opt['train']['fix_flow'] = 25
    # #修正val_freq: 5000为25, opt['val']['val_freq']
    # opt['val']['val_freq'] = 25
    # #修正save_checkpoint_freq: 5000为25, opt['logger']['save_checkpoint_freq']
    # opt['logger']['save_checkpoint_freq'] = 25

    # model_path = opt['path']['pretrain_network_g']  
    model_path = "~"
    opt['path']['pretrain_network_g'] = "./experiments_bak/IART_REDS_N6_300K_Main/models/net_g_245000.pth"
    # 遍历meta_info目录中的所有meta_info_xxx.txt文件  
    meta_info_path = 'meta_info2'
    meta_info_filenames = sorted([os.path.join(meta_info_path, filename) for filename in os.listdir(meta_info_path)  
                     if filename.lower().endswith(('.txt'))])

    for i, meta_info_file in enumerate(meta_info_filenames):
        print("Loop: ", i,"  Begin", "meta info file is:", meta_info_file)
        opt['datasets']['train']['meta_info_file'] = meta_info_file
        train_pipeline(root_path, meta_info_file, model_path, i)
        # opt['datasets']['train']['meta_info_file'] = meta_info_file
        print('current meta_info_file=', opt['datasets']['train']['meta_info_file'])
        
        # del root_path, meta_info_file, i


def mains(yml_dir):
    yml_files = find_files_in_directory(yml_dir)
    for yml_file in yml_files:
        main(yml_file)

if __name__ == "__main__":
    # main()
    current_yml = 'IART_REDS_N6_300K.yml'
    main(current_yml)
