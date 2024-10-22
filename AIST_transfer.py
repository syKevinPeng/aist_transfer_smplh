# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2020 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: Vassilis Choutas, vassilis.choutas@tuebingen.mpg.de

import os
import os.path as osp
from pathlib import Path
import sys
import pickle

sys.path.append(str(Path(__file__).resolve().parent.parent))
import yaml
import numpy as np
import open3d as o3d
import torch
from loguru import logger
from tqdm import tqdm

from smplx import build_layer
from transfermodel.config import parse_args
from transfermodel.data import build_dataloader
from transfermodel.transfer_model import run_fitting
from utils import read_deformation_transfer, np_mesh_to_o3d
import argparse


class SMPLtoSMPLH:
    def __init__(
        self,
        input_folder: str,
        output_folder: str,
        default_yaml_file="/fs/nexus-projects/PhysicsFall/transfermodel/smpl2smplh.yaml",
        verbose=True,
    ):
        self.default_yaml_file = Path(
            default_yaml_file
        )  # this is the default yaml files
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.verbose = verbose
        if self.verbose:
            logger.add(lambda x: tqdm.write(x, end=""), level="INFO", colorize=True)
        else:
            logger.disable("loguru")
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                self.device = torch.device("cuda:0")
                self.device_secondary = torch.device("cuda:1")
                logger.info(
                    f"Using two GPUs: {self.device} and {self.device_secondary}"
                )
            else:
                self.device = torch.device("cuda")
                self.device_secondary = None
                logger.info(f"Using one GPU: {self.device}")
        else:
            raise RuntimeError(
                "No GPU found. Please ensure a GPU is available to run this code."
            )

        if not self.default_yaml_file.exists():
            raise FileNotFoundError(
                f"Default yaml file {self.default_yaml_file} not found"
            )
        if not self.input_folder.exists():
            raise FileNotFoundError(f"Input folder {self.input_folder} not found")
        if not self.output_folder.exists():
            print(f"Creating output directory {self.output_folder}")
            self.output_folder.mkdir(parents=True, exist_ok=True)

        self.cfg = parse_args(self.default_yaml_file)
        self.cfg.output_folder = output_folder
        # cfg.datasets.mesh_folder.data_folder = input_data

    def fitting_one_sequences(self, input_sequence_path, mydevice) -> None:
        # change input path to input_sequence_path
        self.cfg.datasets.mesh_folder.data_folder = input_sequence_path
        file_name = Path(input_sequence_path).stem
        output_folder = self.output_folder / file_name
        logger.info(f"Saving output to: {output_folder}")
        os.makedirs(output_folder, exist_ok=True)

        model_path = self.cfg.body_model.folder
        body_model = build_layer(model_path, **self.cfg.body_model)
        logger.info(body_model)
        body_model = body_model.to(device=mydevice)

        deformation_transfer_path = self.cfg.get("deformation_transfer_path", "")
        def_matrix = read_deformation_transfer(
            deformation_transfer_path, device=mydevice
        )

        # Read mask for valid vertex ids
        mask_ids_fname = osp.expandvars(self.cfg.mask_ids_fname)
        mask_ids = None
        if osp.exists(mask_ids_fname):
            logger.info(f"Loading mask ids from: {mask_ids_fname}")
            mask_ids = np.load(mask_ids_fname)
            mask_ids = torch.from_numpy(mask_ids).to(device=mydevice)
        else:
            logger.warning(f"Mask ids fname not found: {mask_ids_fname}")

        data_obj_dict = build_dataloader(self.cfg)

        dataloader = data_obj_dict["dataloader"]

        for ii, batch in enumerate(tqdm(dataloader)):
            for key in batch:
                if torch.is_tensor(batch[key]):
                    batch[key] = batch[key].to(device=mydevice)
            var_dict = run_fitting(self.cfg, batch, body_model, def_matrix, mask_ids)
            paths = batch["paths"]

            for ii, path in enumerate(paths):
                _, fname = osp.split(path)

                output_path = osp.join(output_folder, f"{osp.splitext(fname)[0]}.pkl")
                with open(output_path, "wb") as f:
                    pickle.dump(var_dict, f)

                output_path = osp.join(output_folder, f"{osp.splitext(fname)[0]}.obj")
                mesh = np_mesh_to_o3d(var_dict["vertices"][ii], var_dict["faces"])
                o3d.io.write_triangle_mesh(output_path, mesh)

    def fitting_a_batch(self, batch_list, device):
        for path in tqdm(batch_list, desc="Processing sequences"):
            self.fitting_one_sequences(path, device)

    def fitting_all_sequences(self, batch_file_list):
        # check if the the items are processed
        for file in batch_file_list:
            if (self.output_folder / file.stem).exists():
                # if the file exists, remove it from the list
                batch_file_list.remove(file)
        if len(batch_file_list) == 0:
            logger.info("All files in the batch already exist. Skipping...")
            return
        # check if two GPUs are available
        if self.device_secondary is not None:
            # split the list into two
            file_list1 = batch_file_list[: len(file_list) // 2]
            file_list2 = batch_file_list[len(file_list) // 2 :]
            # fitting the two lists
            self.fitting_a_batch(file_list1, self.device)
            self.fitting_a_batch(file_list2, self.device_secondary)
        else:
            # get the list of all files
            self.fitting_a_batch(batch_file_list, self.device)


if __name__ == "__main__":
    YAML_FILE = "/fs/nexus-projects/PhysicsFall/transfermodel/smpl2smplh.yaml"
    output_folder = "/fs/nexus-projects/PhysicsFall/data/AIST++/motions-SMPLH"
    input_folder = "/fs/nexus-projects/PhysicsFall/data/AIST++/SMPL_meshes"

    parser = argparse.ArgumentParser(description="Process batch number.")
    parser.add_argument("--batch_num", type=int, help="Batch number to process")
    args = parser.parse_args()

    # check how many split
    file_list = [f for f in Path(input_folder).iterdir() if f.is_dir()].sort()
    batch_size = 15
    num_batches = len(file_list) // batch_size
    print(f"There are {len(file_list)} files")
    print(f"Batch size: {batch_size}, Num batches: {num_batches}")
    print(f"Enter the batch number you want to process (0-{num_batches-1}):")

    batch_num = args.batch_num
    if batch_num < 0 or batch_num >= num_batches:
        print(f"Invalid batch number: {batch_num}")
        exit()
    start_idx = batch_num * batch_size
    end_idx = (batch_num + 1) * batch_size
    file_list = file_list[start_idx:end_idx]

    # check if all files are already processed, if yes, prompt
    # the user to continue
    all_files_exist = True
    for file in file_list:
        if not (Path(output_folder) / file.stem).exists():
            all_files_exist = False
            break
    if all_files_exist:
        raise ValueError("All files in the batch already exist. Exiting.")

    ModelTransfer = SMPLtoSMPLH(
        input_folder=input_folder,
        output_folder=output_folder,
        default_yaml_file=YAML_FILE,
        verbose=False,
    )
    ModelTransfer.fitting_all_sequences(file_list)
