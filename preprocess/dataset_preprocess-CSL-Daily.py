import re
import os
import cv2
cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
import pdb
import glob
import pandas
import argparse
import numpy as np
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool


def csv2dict(dataset_root, anno_path):
    with open(anno_path,'r', encoding='utf-8') as f:
        inputs_list = f.readlines()
    info_dict = dict()
    #info_dict['prefix'] = dataset_root #+ "/fullFrame-210x260px"
    print(f"Generate information dict from {anno_path}")
    for file_idx, file_info in tqdm(enumerate(inputs_list[1:]), total=len(inputs_list)-1):  # Exclude first line
        index, name, length, gloss, char, word, postag = file_info.strip().split("|")
        info_dict[file_idx] = {
            'fileid': name,
            'folder': name+'/*.jpg',
            'signer': 'unknown',
            'label': gloss,
            'num_frames': length,
            'original_info': "|".join(file_info.split("|")[1:]),  #start from 'name', for model inference
        }
    return info_dict


def generate_gt_stm(info, save_path):
    with open(save_path, "w") as f:
        for k, v in info.items():
            if not isinstance(k, int):
                continue
            f.writelines(f"{v['fileid']} 1 {v['signer']} 0.0 1.79769e+308 {v['label']}\n")


def sign_dict_update(total_dict, info):
    for k, v in info.items():
        if not isinstance(k, int):
            continue
        split_label = v['label'].split()
        for gloss in split_label:
            if gloss not in total_dict.keys():
                total_dict[gloss] = 1
            else:
                total_dict[gloss] += 1
    return total_dict


def resize_img(img_path, dsize='210x260px'):
    dsize = tuple(int(res) for res in re.findall("\d+", dsize))
    # img = cv2.imread(img_path)
    img = Image.open(img_path)
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    if img is None:
        print(f'image destroyed: {img_path}, please manually modify the numframes')
        return None
    img = cv2.resize(img, dsize, interpolation=cv2.INTER_LANCZOS4)
    return img


def resize_dataset(video_idx, dsize, info_dict, dataset_root, target_path):
    info = info_dict[video_idx]
    img_list = glob.glob(f"{dataset_root}/{info['folder']}")
    if len(img_list) == len(glob.glob(f"{target_path}/{info['folder']}")):
        return
    for img_path in img_list:
        rs_img = resize_img(img_path, dsize=dsize)
        if rs_img is None:
            info_dict[video_idx]['num_frames'] = info_dict[video_idx]['num_frames']-1
            continue
        rs_img_path = f"{target_path}/{info['fileid']}/{img_path.split('/')[-1]}"
        rs_img_dir = os.path.dirname(rs_img_path)
        if not os.path.exists(rs_img_dir):
            os.makedirs(rs_img_dir)
            cv2.imwrite(rs_img_path, rs_img)
        else:
            cv2.imwrite(rs_img_path, rs_img)


def run_mp_cmd(processes, process_func, process_args):
    with Pool(processes) as p:
        outputs = list(tqdm(p.imap(process_func, process_args), total=len(process_args)))
    return outputs


def run_cmd(func, args):
    return func(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Data process for Visual Alignment Constraint for Continuous Sign Language Recognition.')
    parser.add_argument('--dataset', type=str, default='CSL-Daily',
                        help='save prefix')
    parser.add_argument('--dataset-root', type=str, default='/disk1/dataset/CSL-Daily/sentence/frames_512x512',
                        help='path to the dataset')
    parser.add_argument('--target-path', type=str, default='/disk1/dataset/CSL-Daily_256x256px',
                        help='target path to the dataset')
    parser.add_argument('--annotation-file', type=str, default='video_map.txt',
                        help='annotation file')
    parser.add_argument('--split-file', type=str, default='split_1.txt',
                        help='split file')
    parser.add_argument('--output-res', type=str, default='256x256px',
                        help='resize resolution for image sequence')
    parser.add_argument('--process-image', '-p', action='store_true', default=False,
                        help='resize image')
    parser.add_argument('--multiprocessing', '-m', action='store_true', default=False,
                        help='whether adopts multiprocessing to accelate the preprocess')

    args = parser.parse_args()
    mode = ["train", "dev", "test"]
    sign_dict = dict()
    if not os.path.exists(f"./{args.dataset}"):
        os.makedirs(f"./{args.dataset}")
    
    # generate information dict
    information = csv2dict(args.dataset_root, f"./{args.dataset}/{args.annotation_file}")
    video_index = np.arange(len(information))
    if args.process_image:
        print(f"Resize image to {args.output_res}")
        if args.multiprocessing:
            run_mp_cmd(100, partial(resize_dataset, dsize=args.output_res, info_dict=information, dataset_root=args.dataset_root, target_path=args.target_path), video_index)
        else:
            for idx in tqdm(video_index):
                run_cmd(partial(resize_dataset, dsize=args.output_res, info_dict=information, dataset_root=args.dataset_root, target_path=args.target_path), idx)
                #resize_dataset(idx, dsize=args.output_res, info_dict=information)
    else:
        print("Don't resize images")

    with open(f"./{args.dataset}/{args.split_file}",'r', encoding='utf-8') as f:
        files_list = f.readlines()  
    train_files = []
    dev_files = []
    test_files = []
    for file_idx, file_info in tqdm(enumerate(files_list[1:]), total=len(files_list)-1):  # Exclude first line
        name, split = file_info.strip().split("|")  
        if split == 'train':
            train_files.append(name)
        elif split == 'dev':
            dev_files.append(name)
        elif split == 'test':
            test_files.append(name)
    assert len(train_files) + len(dev_files) + len(test_files) == len(information)
    information_pack = dict()
    for md in mode:
        information_pack[md] = dict()
    train_id = 0
    dev_id = 0
    test_id = 0
    for info_key, info_data in information.items():
        if info_data['fileid'] in train_files:
            information_pack['train'][train_id] = info_data
            train_id += 1
        elif info_data['fileid'] in dev_files:
            information_pack['dev'][dev_id] = info_data
            dev_id +=1
        elif info_data['fileid'] in test_files:
            information_pack['test'][test_id] = info_data
            test_id +=1
        else:
            information_pack['train'][train_id] = info_data    #S000007_P0003_T00
            train_id += 1
    assert len(information_pack['train']) + len(information_pack['dev']) + len(information_pack['test']) == len(information)

    for md in mode:
        np.save(f"./{args.dataset}/{md}_info.npy", information_pack[md])
        # generate groudtruth stm for evaluation
        generate_gt_stm(information_pack[md], f"./{args.dataset}/{args.dataset}-groundtruth-{md}.stm")
    # update the total gloss dict
    sign_dict_update(sign_dict, information)
    sign_dict = sorted(sign_dict.items(), key=lambda d: d[0])
    save_dict = {}
    for idx, (key, value) in enumerate(sign_dict):
        save_dict[key] = [idx + 1, value]
    np.save(f"./{args.dataset}/gloss_dict.npy", save_dict)
