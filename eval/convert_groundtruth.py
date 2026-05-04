# Since the dataset is stored in .mat format, it is easier to covert the file format by Python.
# Also, the transformation is not the main importance task in the project.
# Therefore, I finished the transformation task by Python instead of C++.

import os
import numpy as np
import scipy.io as sio
from PIL import Image
import sys
 
def convert_mat_to_png(mat_path, output_dir):

    mat = sio.loadmat(mat_path)

    groundTruth = mat['groundTruth'][0]
    
    base_name = os.path.splitext(os.path.basename(mat_path))[0]
    
    for idx, gt in enumerate(groundTruth):
        boundary = gt['Boundaries'][0][0]  # Extract boundary array
        
        # Convert to uint8 (0 or 255)
        edge_img = (boundary * 255).astype(np.uint8)
        
        # Save as PNG
        output_path = os.path.join(output_dir, f"{base_name}_gt{idx}.png")
        Image.fromarray(edge_img).save(output_path)
    
    return len(groundTruth)
 
def main():
    if len(sys.argv) > 1:
        bsds_root = sys.argv[1]
    else:
        bsds_root = "./dataset/BSR/BSDS500"
    
    # Paths
    gt_dir = os.path.join(bsds_root, "data/groundTruth/test")
    output_dir = os.path.join(bsds_root, "data/groundTruth/test_png")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all .mat files
    mat_files = [f for f in os.listdir(gt_dir) if f.endswith('.mat')]
    mat_files.sort()
    
    print(f"Converting {len(mat_files)} ground truth files...")
    print(f"From: {gt_dir}")
    print(f"To:   {output_dir}")
    print()
    
    total_annotations = 0
    for mat_file in mat_files:
        mat_path = os.path.join(gt_dir, mat_file)
        num_annotations = convert_mat_to_png(mat_path, output_dir)
        total_annotations += num_annotations
        print(f"✓ {mat_file}: {num_annotations} annotations")
    
    print()
    print(f"Done! Converted {total_annotations} annotations from {len(mat_files)} images")
    print(f"Output: {output_dir}")
 
if __name__ == "__main__":
    main()