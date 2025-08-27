import torch
import numpy as np
import os
import re
from pathlib import Path
from glob import glob
from pymatgen.io.vasp.inputs import Poscar
from torch_geometric.data import Data
from typing import List

# -----------------------------------------------------------------------------
# Configuration and Data Dictionaries
# -----------------------------------------------------------------------------

element_electron_structures = {
    'Cs': [2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 6, 0, 0, 1],
    'Rb': [2, 2, 6, 2, 6, 10, 2, 6, 0, 1, 0, 0, 0, 0],
    'Ti': [2, 2, 6, 2, 6, 2, 2, 0, 0, 0, 0, 0, 0, 0],
    'Zr': [2, 2, 6, 2, 6, 2, 2, 0, 0, 0, 0, 0, 0, 0],
    'Pd': [2, 2, 6, 2, 6, 10, 2, 6, 10, 0, 0, 0, 0, 0],
    'Sn': [2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 2, 0, 0, 0],
    'Te': [2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 4, 0, 0, 0],
    'Hf': [2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 6, 14, 2, 2],
    'Pt': [2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 6, 14, 9, 1],
    'Cl': [2, 2, 6, 2, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Br': [2, 2, 6, 2, 6, 10, 2, 5, 0, 0, 0, 0, 0, 0],
    'I':  [2, 2, 6, 2, 6, 10, 2, 6, 10, 2, 5, 0, 0, 0]
}

element_electronegativity = {
    'Cs': [0.659], 'Rb': [0.706], 'Ti': [1.38], 'Zr': [1.32], 'Pd': [1.58],
    'Sn': [1.824], 'Te': [2.158], 'Hf': [1.16], 'Pt': [1.72], 'Cl': [2.869],
    'Br': [2.685], 'I': [2.359]
}

element_ionic_radius = {
    'Cs': [1.67], 'Rb': [1.52], 'Ti': [0.605], 'Zr': [0.72], 'Pd': [0.615],
    'Sn': [0.69], 'Te': [0.52], 'Hf': [0.58], 'Pt': [0.625], 'Cl': [1.81],
    'Br': [1.96], 'I': [2.2]
}

first_group_elements = set(['Cs', 'Rb'])
second_group_elements = set(['Ti', 'Zr', 'Pd', 'Sn', 'Te', 'Hf', 'Pt'])
third_group_elements = set(['Cl', 'Br', 'I'])

root_dir = "./Data"

# -----------------------------------------------------------------------------
# Main Processing Logic
# -----------------------------------------------------------------------------

def process_poscar_to_data(poscar_path: Path) -> Data | None:
    try:
        poscar = Poscar.from_file(poscar_path)
        structure = poscar.structure
        lattice = structure.lattice.matrix

        atom_features = []
        positions = []
        frac_positions = []
        for site in structure:
            ele = site.specie

            electronic_structure = element_electron_structures.get(ele.symbol)
            electronegativity = element_electronegativity.get(ele.symbol)
            ionic_radius = element_ionic_radius.get(ele.symbol)

            if electronic_structure is None or electronegativity is None or ionic_radius is None:
                print(f"Warning: Missing feature data for element {ele.symbol} in {poscar_path}. Skipping file.")
                return None

            group_encoding = [0, 0, 0]
            if ele.symbol in first_group_elements:
                group_encoding[0] = 1
            elif ele.symbol in second_group_elements:
                group_encoding[1] = 1
            elif ele.symbol in third_group_elements:
                group_encoding[2] = 1

            feature_vector = electronic_structure + electronegativity + ionic_radius + group_encoding
            atom_features.append(feature_vector)
            positions.append(site.coords)
            frac_positions.append(site.frac_coords) 

        edge_indices = []
        coord_vectors = []

        for idx, site in enumerate(structure):
            neighbors_r = 0
            if site.specie.symbol in first_group_elements or site.specie.symbol in third_group_elements:
                neighbors_r = 4.2
            elif site.specie.symbol in second_group_elements:
                neighbors_r = 3.0
            else:
                continue

            neighbors = structure.get_neighbors(site, r=neighbors_r, include_index=True)
            if not neighbors:
                continue

            for neighbor in neighbors:
                neighbor_site = neighbor[0]
                neighbor_index = neighbor[2]
                if neighbor_index == idx:
                    continue

                if site.specie.symbol in third_group_elements and neighbor_site.specie.symbol in third_group_elements:
                    continue

                image = neighbor[3]
                unwrapped_frac_coords = structure[neighbor_index].frac_coords + image
                vector = unwrapped_frac_coords - site.frac_coords
                cart_vector = np.dot(vector, lattice)

                edge_indices.append([idx, neighbor_index])
                coord_vectors.append(cart_vector)

        if not atom_features or not edge_indices:
            print(f"Warning: No valid atoms or edges generated for {poscar_path}. Skipping.")
            return None

        # ✅ 统一批量转换为 numpy 再转 torch
        atom_features_np = np.array(atom_features, dtype=np.float32)
        positions_np = np.array(positions, dtype=np.float32)
        frac_positions_np = np.array(frac_positions, dtype=np.float32)
        edge_indices_np = np.array(edge_indices, dtype=np.int64).T
        coord_vectors_np = np.array(coord_vectors, dtype=np.float32)

        data_object = Data(
            x=torch.from_numpy(atom_features_np),
            edge_index=torch.from_numpy(edge_indices_np).contiguous(),
            edge_vec=torch.from_numpy(coord_vectors_np),
            pos=torch.from_numpy(positions_np),
            pos_frac=torch.from_numpy(frac_positions_np)
        )

        edge_dim = 40 
        rc = torch.linspace(0.0, 4.5, edge_dim)
        edgenorm = torch.norm(data_object.edge_vec, dim=1, keepdim=False)
        average_edgenorm = torch.mean(edgenorm)
        if average_edgenorm > 1e-6:
            edgenorm = edgenorm / average_edgenorm
            data_object.edge_vec = data_object.edge_vec / average_edgenorm

        data_object.edge_attr = torch.exp(-(edgenorm.unsqueeze(-1) - rc.unsqueeze(0))**2 / 0.1)

        return data_object

    except Exception as e:
        print(f"Error processing {poscar_path}: {e}")
        return None

def main():
    all_sequences: List[List[Data]] = []
    root_path = Path(root_dir)
    if not root_path.is_dir():
        print(f"Error: Root directory not found at {root_path}")
        return

    print("Starting data preprocessing...")
    for subfolder_path in root_path.iterdir():
        if subfolder_path.is_dir():
            print(f"Processing folder: {subfolder_path.name}")
            all_poscar_files_in_folder = glob(str(subfolder_path / "POSCAR*"))

            poscar_paths = []
            for p_path in all_poscar_files_in_folder:
                match = re.search(r'POSCAR_(\d+)', p_path)
                if match:
                    poscar_paths.append((int(match.group(1)), p_path))

            poscar_paths.sort(key=lambda x: x[0])
            poscar_paths = [p[1] for p in poscar_paths]

            if not poscar_paths:
                print(f"Warning: No POSCAR_X files found in {subfolder_path.name}. Skipping this folder.")
                continue

            current_sequence = []
            for poscar_path_str in poscar_paths:
                poscar_path = Path(poscar_path_str)
                data_object = process_poscar_to_data(poscar_path)
                if isinstance(data_object, Data):
                    current_sequence.append(data_object)

            if len(current_sequence) > 1: 
                all_sequences.append(current_sequence)
            else:
                print(f"Warning: Sequence for {subfolder_path.name} has less than 2 valid steps, skipping.")
                
    print(f"\nFinished preprocessing. Found {len(all_sequences)} sequences.")
    final_sequences_to_save = []
    for seq in all_sequences:
        if all(isinstance(item, Data) for item in seq):
            final_sequences_to_save.append(seq)
        else:
            print("Warning: A sequence contains non-Data objects and will not be saved.")

    torch.save(final_sequences_to_save, 'all_sequences.pt')
    print(f"All {len(final_sequences_to_save)} valid sequences saved to 'all_sequences.pt'.")

if __name__ == "__main__":
    main()
