
import argparse
import os
import sys
import numpy as np
import torch
from training_utils import load_config
from utils import floor_plan_from_scene

from scene_synthesis.datasets import filter_function, get_dataset_raw_and_encoded
from scene_synthesis.networks import build_network

def main(argv):
    parser = argparse.ArgumentParser(
        description="Generate scenes using a previously trained model (Boxes only)"
    )

    parser.add_argument(
        "config_file",
        help="Path to the file that contains the experiment configuration"
    )
    parser.add_argument(
        "--split_file",
        default="../config/houzz_bathroom_inference_splits.csv",
        help="The split file to be used for inference"
    )
    parser.add_argument(
        "output_directory",
        default="/tmp/",
        help="Path to the output directory"
    )
    parser.add_argument(
        "--weight_file",
        default=None,
        help="Path to a pretrained model"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="The number of samples to generate for each scene"
    )
    parser.add_argument(
        "--path_to_floor_plan_textures",
        default="../demo/floor_plan_texture_images",
        help="Path to floor texture images"
    )
    parser.add_argument(
        "--no_texture",
        action="store_true",
        help="if remove the texture"
    )
    parser.add_argument(
        "--clip_denoised",
        action="store_true",
        help="if clip_denoised"
    )
    parser.add_argument(
        "--scene_id",
        default=None,
        help="The scene id to be used for conditioning"
    )
    parser.add_argument(
        "--ddim",
        action="store_true",
        help="if use ddim"
    )

    args = parser.parse_args(argv)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print("Running code on", device)

    # Check if output directory exists and if it doesn't create it
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    config = load_config(args.config_file)
    # Override for inference on specific small split
    config["data"]["annotation_file"] = args.split_file
    config["validation"]["splits"] = ["train", "test"]

    ########## make it for evaluation
    if 'text' in config["data"]["encoding_type"]:
        if 'textfix' not in config["data"]["encoding_type"]:
            config["data"]["encoding_type"] = config["data"]["encoding_type"].replace('text', 'textfix')

    if "no_prm" not in config["data"]["encoding_type"]:
        print('NO PERM AUG in test')
        config["data"]["encoding_type"] = config["data"]["encoding_type"] + "_no_prm"
    print('encoding type :', config["data"]["encoding_type"])
    ####### 

    raw_dataset, dataset = get_dataset_raw_and_encoded(
        config["data"],
        filter_fn=filter_function(
            config["data"],
            split=config["validation"].get("splits", ["train", "test"])
        ),
        split=config["validation"].get("splits", ["train", "test"])
    )


    print("Loaded {} scenes with {} object types:".format(
        len(dataset), dataset.n_object_types)
    )
    network, _, _ = build_network(
        dataset.feature_size, dataset.n_classes,
        config, args.weight_file, device=device
    )

    network.eval()

    given_scene_id = None
    if args.scene_id:
        given_scene_id = int(args.scene_id) if args.scene_id.isdigit() else None
        if given_scene_id is not None:
            print(f"Using fixed scene at index {given_scene_id}")

    print(f"Generating samples for {len(dataset)} sequences...")
    import shutil
    
    for i in range(len(dataset)):
        scene_idx = given_scene_id if given_scene_id is not None else (i % len(dataset))
        samples = dataset[scene_idx]
        
        try:
            actual_scene_id = raw_dataset._tags[scene_idx]
        except Exception as e:
            actual_scene_id = f"scene_{scene_idx}"
        
        print(f"{i+1} / {len(dataset)}: Processing scene {actual_scene_id}")
        
        # Prepare conditioned partial boxes
        room_layout_size = config["data"].get("room_layout_size", "64,64").split(",")
        room_layout_h, room_layout_w = int(room_layout_size[0]), int(room_layout_size[1])
        room_mask = torch.zeros((1, 1, room_layout_h, room_layout_w), device=device)
        
        partial_num_points = config["network"].get("partial_num_points", 40)
        translations = samples['translations'][:partial_num_points]
        sizes = samples['sizes'][:partial_num_points]
        angles = samples['angles'][:partial_num_points]
        class_labels = samples['class_labels'][:partial_num_points]
        
        comp_tensor = lambda x: torch.from_numpy(x).float() if not isinstance(x, torch.Tensor) else x
        partial_boxes = torch.cat([
            comp_tensor(translations),
            comp_tensor(sizes),
            comp_tensor(angles),
            comp_tensor(class_labels)
        ], dim=-1).unsqueeze(0).to(device) # (1, 40, 17)

        # Output directory
        sample_dir = os.path.join(args.output_directory, actual_scene_id)
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)

        # Generate multiple samples sequentially
        for j in range(args.n_samples):
            print(f"  Generating sample {j}...")
            with torch.no_grad():
                # Use complete_scene (sequential is safer for the network as implemented)
                bbox_params = network.complete_scene(
                    room_mask=room_mask,
                    num_points=config["network"]["sample_num_points"],
                    point_dim=config["network"]["point_dim"],
                    partial_boxes=partial_boxes,
                    batch_size=1,
                    device=device,
                    clip_denoised=args.clip_denoised,
                    batch_seeds=torch.tensor([i * args.n_samples + j], device=device),
                    ddim=args.ddim
                )
                
            boxes = dataset.post_process(bbox_params)
            
            sample_boxes = {}
            for k, v in boxes.items():
                if isinstance(v, torch.Tensor):
                    sample_boxes[k] = v[0].cpu().numpy()
                else:
                    sample_boxes[k] = v[0]
            
            npz_path = os.path.join(sample_dir, f"boxes_{j}.npz")
            np.savez(npz_path, **sample_boxes)
            print(f"    Saved {npz_path}")

        # Copy assets
        if "_simple_design_filtered" in actual_scene_id:
            source_folder_name = actual_scene_id.replace("_simple_design_filtered", "")
            source_dir = os.path.join(config["data"]["dataset_directory"], "..", "bathroom_svg_files", source_folder_name)
            for filename in ["simple_design_filtered.json", "simple_design_label_2.svg", "simple_design.json"]:
                source_file = os.path.join(source_dir, filename)
                if os.path.exists(source_file):
                    shutil.copy2(source_file, os.path.join(sample_dir, filename))
                    
    print("Done generation.")

if __name__ == "__main__":
    main(sys.argv[1:])
