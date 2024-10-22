# Use this script to load all AIST data and convert it to SMPL meshes
import torch
import lightning as pl
import os, glob, sys
from torch.utils.data import DataLoader
import numpy as np
import multiprocessing as mp
import pickle
from tqdm import tqdm, trange
import smplx
from pathlib import Path
import trimesh
import re

HUMAN_MODEL_PATH = "/fs/nexus-projects/PhysicsFall/data/smpl/models"
DEVICE = "cuda"


class AistMeshGeneartor:
    def __init__(
        self,
        data_dir,
        mesh_output_dir,
        valid_list_file,
        batch_size=64,
        train_val_test_split=[0.8, 0.2, 0],
        num_workers=8,
        fps=30,
        force_preprocess=False,
        verbose=False,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.data_output_dir = Path(mesh_output_dir)
        self.force_preprocess = force_preprocess
        self.num_workers = num_workers
        self.train_val_test_split = train_val_test_split
        self.batch_size = batch_size
        self.verbose = verbose
        self.dataset_fps = 60
        self.fps = fps
        self.valid_list_file = Path(valid_list_file)
        # sanity check
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory {self.data_dir} not found")
        if not self.data_output_dir.exists():
            print(f"Creating output directory {self.data_output_dir}")
            self.data_output_dir.mkdir(parents=True, exist_ok=True)
        if not self.valid_list_file.exists():
            raise FileNotFoundError(f"Valid list file {self.valid_list_file} not found")

    # convert AIST SMPL parameters into meshes
    def save_one_file_to_mesh(self, file_path):
        # if self.verbose:
        #     print(f"Processing file {file_path}")
        data = np.load(file_path, allow_pickle=True)
        file_name = Path(file_path).stem
        if not (self.data_output_dir / file_name).exists():
            Path(self.data_output_dir / file_name).mkdir(parents=True, exist_ok=True)
        else:
            print(f"Existing {file_name}. Skipping...")
            return
        smpl_poses = data["smpl_poses"]  # (N, 24x3)
        smpl_scaling = data["smpl_scaling"]  # ( 1)
        smpl_trans = data["smpl_trans"]  # (N, 3)
        # subsample
        bin_len = self.dataset_fps / float(self.fps)
        length_up = int(smpl_poses.shape[0] / bin_len)
        tt = (bin_len * np.arange(0, length_up)).astype(np.int32).tolist()
        pose = torch.from_numpy(smpl_poses[tt]).float()
        trans = torch.from_numpy(smpl_trans[tt]).float()
        scale = torch.from_numpy(smpl_scaling.reshape(1, 1)).float()
        root_orient = pose[:, :3]
        body = pose[:, 3:]
        # get the mesh
        model = smplx.create(model_path=HUMAN_MODEL_PATH, model_type="smpl")
        output = model.forward(
            global_orient=root_orient,  # type:ignore
            body_pose=body,  # type:ignore
            transl=trans,  # type:ignore
            scaling=scale,  # type:ignore
        )
        vertices = output.vertices.detach().cpu().numpy().squeeze()
        joints = output.joints.detach().cpu().numpy().squeeze()
        vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
        for pose_idx in trange(body.size(0)):
            tri_mesh = trimesh.Trimesh(
                vertices[pose_idx],
                model.faces,
                vertex_colors=vertex_colors,
                process=False,
            )
            output_path = (
                self.data_output_dir / file_name / "{0:04d}.obj".format(pose_idx)
            )
            tri_mesh.export(str(output_path))

    def extract_valid_files(self):
        """
        Extract the list of valid files namesfrom the valid list file
        """
        with open(self.valid_list_file, "r") as f:
            valid_list = f.readlines()
        valid_list = [x.strip() for x in valid_list]
        # get only the files names from the URL
        valid_list = [Path(x).stem for x in valid_list]
        # replacing the cXX (camera number) to cALL with regex
        valid_list = [re.sub(r"_c\d{2}_", "_cAll_", x) for x in valid_list]
        # check if the file exists in the data directory
        motion_list = [self.data_dir / f"{x}.pkl" for x in valid_list if (self.data_dir / f"{x}.pkl").exists()]
        if self.verbose:
            print(
                f"Total valid files: {len(valid_list)};Total files founded: {len(motion_list)}"
            )

        return motion_list

    def process_files_to_meshes(self, all_data_files_list, num_workers):
        """
        Use Multiprocessing to process the SMPL parameters to meshes
        """
        with mp.Pool(num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(
                        self.save_one_file_to_mesh, all_data_files_list
                    ),
                    total=len(all_data_files_list),
                )
            )
        return results

    def convert_meshes(self) -> None:
        all_data_files_list = self.extract_valid_files()
        print(f"Number of valid files: {len(all_data_files_list)}")
        # debugging
        # self.process_one_file(all_data_files_list[1])
        self.process_files_to_meshes(all_data_files_list, self.num_workers)


if __name__ == "__main__":
    # data module testing
    data_dir = "/fs/nexus-projects/PhysicsFall/data/AIST++/motions-SMPL"
    output_data_dir = "/fs/nexus-projects/PhysicsFall/data/AIST++/SMPL_meshes"
    valid_list_file = "/fs/nexus-projects/PhysicsFall/data/AIST++/video_list.txt"

    subsampling_fps = 10
    generator = AistMeshGeneartor(
        data_dir,
        output_data_dir,
        valid_list_file=valid_list_file,
        fps=subsampling_fps,
        verbose=True,
    )
    generator.convert_meshes()
