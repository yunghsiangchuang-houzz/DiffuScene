
import argparse
import os
import sys
import numpy as np
import torch
from tqdm import tqdm
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
        "--use_text_description",
        action="store_true",
        help="if use text description"
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
    parser.add_argument(
        "--single_inference",
        action="store_true",
        help="Use single (sequential) inference instead of batch inference (slower but uses less memory)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Total batch size for inference. Processes floor(batch_size/n_samples) scenes per batch. "
             "E.g., --batch_size=128 --n_samples=6 -> 21 scenes x 6 samples = 126 per inference"
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
    # config["data"]["annotation_file"] = args.split_file
    # config["validation"]["splits"] = ["train", "test"]

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
            split=["test"]
            # split=["train"]

        ),
        split=["test"]
        # split=["train"]
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

    # Batch size mode: process multiple scenes together with controlled batch size
    if args.batch_size is not None:
        scenes_per_batch = args.batch_size // args.n_samples
        if scenes_per_batch < 1:
            print(f"Warning: batch_size ({args.batch_size}) < n_samples ({args.n_samples}). Setting scenes_per_batch=1")
            scenes_per_batch = 1
        
        actual_batch_size = scenes_per_batch * args.n_samples
        num_batches = (len(dataset) + scenes_per_batch - 1) // scenes_per_batch  # ceil division
        
        print(f"Using batch mode: {len(dataset)} scenes x {args.n_samples} samples")
        print(f"  -> {scenes_per_batch} scenes per batch x {args.n_samples} samples = {actual_batch_size} per inference")
        print(f"  -> {num_batches} total inference calls")
        
        # Preload all scene data
        room_layout_size = config["data"].get("room_layout_size", "64,64").split(",")
        room_layout_h, room_layout_w = int(room_layout_size[0]), int(room_layout_size[1])
        partial_num_points = config["network"].get("partial_num_points", 40)
        
        all_partial_boxes = []
        all_scene_ids = []
        all_sample_dirs = []
        
        print("Preloading all scene data...")
        for i in tqdm(range(len(dataset)), desc="Loading scenes", unit="scene"):
            scene_idx = given_scene_id if given_scene_id is not None else i
            samples = dataset[scene_idx]
            
            try:
                actual_scene_id = raw_dataset._tags[scene_idx]
            except Exception:
                actual_scene_id = f"scene_{scene_idx}"
            
            all_scene_ids.append(actual_scene_id)
            
            # Prepare output directory
            sample_dir = os.path.join(args.output_directory, actual_scene_id)
            if not os.path.exists(sample_dir):
                os.makedirs(sample_dir)
            all_sample_dirs.append(sample_dir)
            
            # Prepare partial boxes
            translations = samples['translations'][:partial_num_points]
            sizes = samples['sizes'][:partial_num_points]
            angles = samples['angles'][:partial_num_points]
            class_labels = samples['class_labels'][:partial_num_points]
            
            comp_tensor = lambda x: torch.from_numpy(x).float() if not isinstance(x, torch.Tensor) else x
            partial_box = torch.cat([
                comp_tensor(translations),
                comp_tensor(sizes),
                comp_tensor(angles),
                comp_tensor(class_labels)
            ], dim=-1)  # (num_partial, 17)
            all_partial_boxes.append(partial_box)
            
            # Copy assets
            if "_simple_design_filtered" in actual_scene_id:
                source_folder_name = actual_scene_id.replace("_simple_design_filtered", "")
                source_dir = os.path.join(config["data"]["dataset_directory"], "..", source_folder_name)
                for filename in ["simple_design_filtered.json", "simple_design_label.svg", "simple_design.json"]:
                    source_file = os.path.join(source_dir, filename)
                    if os.path.exists(source_file):
                        shutil.copy2(source_file, os.path.join(sample_dir, filename))
        
        # Text description (same for all if used)
        if args.use_text_description:
            text_description = "The bathroom has one vanity, one toilet, and one tub. "
        else:
            text_description = None
        
        # Process scenes in batches
        num_scenes = len(dataset)
        for batch_idx in tqdm(range(num_batches), desc="Processing batches", unit="batch"):
            start_idx = batch_idx * scenes_per_batch
            end_idx = min(start_idx + scenes_per_batch, num_scenes)
            batch_scenes = end_idx - start_idx
            
            print(f"\nBatch {batch_idx + 1}/{num_batches}: scenes {start_idx}-{end_idx-1} ({batch_scenes} scenes x {args.n_samples} samples)")
            
            # Stack partial boxes for this batch of scenes
            batch_partial_boxes = torch.stack(all_partial_boxes[start_idx:end_idx], dim=0)  # (batch_scenes, num_partial, 17)
            
            # Repeat each scene n_samples times: (batch_scenes * n_samples, num_partial, 17)
            # Layout: [scene0_sample0, scene0_sample1, ..., scene0_sampleN, scene1_sample0, ...]
            batch_partial_boxes = batch_partial_boxes.unsqueeze(1).repeat(1, args.n_samples, 1, 1)  # (batch_scenes, n_samples, num_partial, 17)
            batch_partial_boxes = batch_partial_boxes.view(-1, batch_partial_boxes.shape[2], batch_partial_boxes.shape[3]).to(device)  # (batch_scenes * n_samples, num_partial, 17)
            
            # Create room mask for this batch
            total_samples = batch_scenes * args.n_samples
            room_mask_batch = torch.zeros((total_samples, 1, room_layout_h, room_layout_w), device=device)
            
            # Create batch seeds: unique seed for each (scene, sample) pair
            # seed = scene_idx * n_samples + sample_idx
            batch_seeds = []
            for scene_offset in range(batch_scenes):
                scene_idx = start_idx + scene_offset
                for sample_idx in range(args.n_samples):
                    batch_seeds.append(scene_idx * args.n_samples + sample_idx)
            batch_seeds = torch.tensor(batch_seeds, device=device)
            
            with torch.no_grad():
                bbox_params = network.complete_scene(
                    room_mask=room_mask_batch,
                    num_points=config["network"]["sample_num_points"],
                    point_dim=config["network"]["point_dim"],
                    partial_boxes=batch_partial_boxes,
                    text=text_description,
                    batch_size=total_samples,
                    device=device,
                    clip_denoised=args.clip_denoised,
                    batch_seeds=batch_seeds,
                    ddim=args.ddim,
                    keep_empty=True
                )
            
            boxes = dataset.post_process(bbox_params)
            
            # Save results: layout is [scene0_sample0, scene0_sample1, ..., scene1_sample0, ...]
            for scene_offset in range(batch_scenes):
                scene_idx = start_idx + scene_offset
                sample_dir = all_sample_dirs[scene_idx]
                
                for sample_idx in range(args.n_samples):
                    result_idx = scene_offset * args.n_samples + sample_idx
                    
                    sample_boxes = {}
                    for k, v in boxes.items():
                        if isinstance(v, torch.Tensor):
                            sample_boxes[k] = v[result_idx].cpu().numpy()
                        else:
                            sample_boxes[k] = v[result_idx]
                    
                    npz_path = os.path.join(sample_dir, f"boxes_{sample_idx}.npz")
                    np.savez(npz_path, **sample_boxes)
        
        print("Done generation.")
        return
    
    # Original per-scene loop (when --batch_size is not specified)
    for i in tqdm(range(len(dataset)), desc="Generating scenes", unit="scene"):
        scene_idx = given_scene_id if given_scene_id is not None else (i % len(dataset))
        samples = dataset[scene_idx]
        
        try:
            actual_scene_id = raw_dataset._tags[scene_idx]
        except Exception as e:
            actual_scene_id = f"scene_{scene_idx}"
        
        tqdm.write(f"{i+1} / {len(dataset)}: Processing scene {actual_scene_id}")
        
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

        if args.use_text_description:
            text_description = (
                "The bathroom has one vanity, one toilet, and one tub. "
            )
        else:
            text_description = None

        tqdm.write(f"Text description: {text_description}")

        if args.single_inference:
            # Generate multiple samples SEQUENTIALLY (one at a time)
            tqdm.write(f"  Generating {args.n_samples} samples sequentially...")
            for j in range(args.n_samples):
                tqdm.write(f"    Generating sample {j}...")
                with torch.no_grad():
                    bbox_params = network.complete_scene(
                        room_mask=room_mask,
                        num_points=config["network"]["sample_num_points"],
                        point_dim=config["network"]["point_dim"],
                        partial_boxes=partial_boxes,
                        text=text_description,
                        batch_size=1,
                        device=device,
                        clip_denoised=args.clip_denoised,
                        batch_seeds=torch.tensor([i * args.n_samples + j], device=device),
                        ddim=args.ddim
                    )
                
                boxes = dataset.post_process(bbox_params)
                
                # Save this sample
                sample_boxes = {}
                for k, v in boxes.items():
                    if isinstance(v, torch.Tensor):
                        sample_boxes[k] = v[0].cpu().numpy()
                    else:
                        sample_boxes[k] = v[0]
                
                npz_path = os.path.join(sample_dir, f"boxes_{j}.npz")
                np.savez(npz_path, **sample_boxes)
            
            tqdm.write(f"  Saved {args.n_samples} samples to {sample_dir}")
        else:
            # Generate multiple samples in a BATCH for faster inference
            tqdm.write(f"  Generating {args.n_samples} samples in batch...")
            with torch.no_grad():
                # Expand inputs for batch processing
                room_mask_batch = room_mask.expand(args.n_samples, -1, -1, -1)  # (n_samples, 1, H, W)
                partial_boxes_batch = partial_boxes.expand(args.n_samples, -1, -1)  # (n_samples, num_partial, 17)
                
                # Create batch seeds for reproducibility
                batch_seeds = torch.arange(i * args.n_samples, (i + 1) * args.n_samples, device=device)
                
                bbox_params = network.complete_scene(
                    room_mask=room_mask_batch,
                    num_points=config["network"]["sample_num_points"],
                    point_dim=config["network"]["point_dim"],
                    partial_boxes=partial_boxes_batch,
                    text=text_description,
                    batch_size=args.n_samples,  # Generate all samples in one batch
                    device=device,
                    clip_denoised=args.clip_denoised,
                    batch_seeds=batch_seeds,
                    ddim=args.ddim,
                    keep_empty=True
                )
            
            boxes = dataset.post_process(bbox_params)
            
            # Save each sample from the batch
            for j in range(args.n_samples):
                sample_boxes = {}
                for k, v in boxes.items():
                    if isinstance(v, torch.Tensor):
                        sample_boxes[k] = v[j].cpu().numpy()
                    else:
                        sample_boxes[k] = v[j]
                
                npz_path = os.path.join(sample_dir, f"boxes_{j}.npz")
                np.savez(npz_path, **sample_boxes)
            
            tqdm.write(f"  Saved {args.n_samples} samples to {sample_dir}")

        # Copy assets
        if "_simple_design_filtered" in actual_scene_id:
            source_folder_name = actual_scene_id.replace("_simple_design_filtered", "")
            source_dir = os.path.join(config["data"]["dataset_directory"], "..", source_folder_name)
            for filename in ["simple_design_filtered.json", "simple_design_label.svg", "simple_design.json"]:
                source_file = os.path.join(source_dir, filename)
                if os.path.exists(source_file):
                    shutil.copy2(source_file, os.path.join(sample_dir, filename))
                    
    print("Done generation.")

if __name__ == "__main__":
    main(sys.argv[1:])
