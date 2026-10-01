import json
import argparse
import math
import os
import numpy as np
import glob
import csv
import shutil
import tempfile
import cv2
import torch
import subprocess
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf


from torchvision.io import read_video
# from decord import VideoReader, cpu  # unused; decord not installed in five-bench env
import imageio

from metrics_calculator import MetricsCalculator, average_niqe_from_txt


def mask_decode(encoded_mask, image_shape=[512,512]):
    length = image_shape[0] * image_shape[1]
    mask_array = np.zeros((length,))
    
    for i in range(0, len(encoded_mask), 2):
        splice_len = min(encoded_mask[i+1], length-encoded_mask[i])
        for j in range(splice_len):
            mask_array[encoded_mask[i]+j]=1
            
    mask_array = mask_array.reshape(image_shape[0], image_shape[1])
    # to avoid annotation errors in boundary
    mask_array[0,:]=1
    mask_array[-1,:]=1
    mask_array[:,0]=1
    mask_array[:,-1]=1
            
    return mask_array



#✨ R28: background-preservation metrics scored on the complement of a mask UNION.
# Everything below with an `_unedit_part` suffix uses `1 - src_mask`, and the call site
# (see the frame loop) passes the SOURCE mask as `tgt_mask` too, so "background" is
# whatever the source object did not occupy. An edit that moves, grows or reshapes the
# object puts its new pixels inside that region, where they are scored as failed
# preservation -- penalising correct edits and rewarding under-editing.
#
# `_unedit_union`       -> 1 - (M_src | M_tgt_arm)   this arm's own target mask
# `_unedit_union_fixed` -> 1 - (M_src | U_arms M_tgt) one fixed region for every arm
#
# Both are supersets of M_src, so both backgrounds are subsets of today's `1 - M_src`.
_UNION_BASES = {
    "psnr": "calculate_psnr",
    "lpips": "calculate_lpips",
    "mse": "calculate_mse",
    "ssim": "calculate_ssim",
    "structure_distance": "calculate_structure_distance",
}


def _union_metric(metrics_calculator, metric, src_image, tgt_image, union):
    """Shared body for the two union families. Returns 'nan' when unusable."""
    if union is None:
        return "nan"
    suffix = "_unedit_union_fixed" if metric.endswith("_unedit_union_fixed") else "_unedit_union"
    base = metric[: -len(suffix)]
    fn_name = _UNION_BASES.get(base)
    if fn_name is None:
        return "nan"
    bg = 1 - union
    if bg.sum() == 0:
        # The union covers the frame; there is no background left to score.
        return "nan"
    # Same call shape as `_unedit_part`: multiply-by-mask, not crop, so the families
    # differ ONLY by which region is excluded.
    return getattr(metrics_calculator, fn_name)(src_image, tgt_image, bg, bg)


def calculate_metric(metrics_calculator, metric, src_image, tgt_image, src_mask, tgt_mask,src_prompt,tgt_prompt,
                     src_image_path, tgt_image_path, src_save_file_niqe, tgt_save_file_niqe,
                     tgt_mask_real=None, fixed_union=None):
    #✨ R28: dispatched before the existing branches; every name below is new, so no
    # existing metric changes behaviour. `tgt_mask_real is None` => 'nan', which is what
    # a run without --tgt_mask_dir gets.
    if metric.endswith("_unedit_union_fixed"):
        return _union_metric(metrics_calculator, metric, src_image, tgt_image, fixed_union)
    if metric.endswith("_unedit_union"):
        union = None
        if tgt_mask_real is not None:
            union = np.clip(src_mask.astype(np.int32) + tgt_mask_real.astype(np.int32), 0, 1)
        return _union_metric(metrics_calculator, metric, src_image, tgt_image, union)
    if metric=="psnr":
        return metrics_calculator.calculate_psnr(src_image, tgt_image, None, None)
    if metric=="lpips":
        return metrics_calculator.calculate_lpips(src_image, tgt_image, None, None)
    if metric=="mse":
        return metrics_calculator.calculate_mse(src_image, tgt_image, None, None)
    if metric=="ssim":
        return metrics_calculator.calculate_ssim(src_image, tgt_image, None, None)
    if metric=="structure_distance":
        return metrics_calculator.calculate_structure_distance(src_image, tgt_image, None, None)
    if metric=="psnr_unedit_part":
        if (1-src_mask).sum()==0 or (1-tgt_mask).sum()==0:
            return "nan"
        else:
            return metrics_calculator.calculate_psnr(src_image, tgt_image, 1-src_mask, 1-tgt_mask)
    if metric=="lpips_unedit_part":
        if (1-src_mask).sum()==0 or (1-tgt_mask).sum()==0:
            return "nan"
        else:
            return metrics_calculator.calculate_lpips(src_image, tgt_image, 1-src_mask, 1-tgt_mask)
    if metric=="mse_unedit_part":
        if (1-src_mask).sum()==0 or (1-tgt_mask).sum()==0:
            return "nan"
        else:
            return metrics_calculator.calculate_mse(src_image, tgt_image, 1-src_mask, 1-tgt_mask)
    if metric=="ssim_unedit_part":
        if (1-src_mask).sum()==0 or (1-tgt_mask).sum()==0:
            return "nan"
        else:
            return metrics_calculator.calculate_ssim(src_image, tgt_image, 1-src_mask, 1-tgt_mask)
    if metric=="structure_distance_unedit_part":
        if (1-src_mask).sum()==0 or (1-tgt_mask).sum()==0:
            return "nan"
        else:
            return metrics_calculator.calculate_structure_distance(src_image, tgt_image, 1-src_mask, 1-tgt_mask)
    if metric=="psnr_edit_part":
        if src_mask.sum()==0 or tgt_mask.sum()==0:
            return "nan"
        else:
            return metrics_calculator.calculate_psnr(src_image, tgt_image, src_mask, tgt_mask)
    if metric=="lpips_edit_part":
        if src_mask.sum()==0 or tgt_mask.sum()==0:
            return "nan"
        else:
            return metrics_calculator.calculate_lpips(src_image, tgt_image, src_mask, tgt_mask)
    if metric=="mse_edit_part":
        if src_mask.sum()==0 or tgt_mask.sum()==0:
            return "nan"
        else:
            return metrics_calculator.calculate_mse(src_image, tgt_image, src_mask, tgt_mask)
    if metric=="ssim_edit_part":
        if src_mask.sum()==0 or tgt_mask.sum()==0:
            return "nan"
        else:
            return metrics_calculator.calculate_ssim(src_image, tgt_image, src_mask, tgt_mask)
    if metric=="structure_distance_edit_part":
        if src_mask.sum()==0 or tgt_mask.sum()==0:
            return "nan"
        else:
            return metrics_calculator.calculate_structure_distance(src_image, tgt_image, src_mask, tgt_mask)
    if metric=="clip_similarity_source_image":
        return metrics_calculator.calculate_clip_similarity(src_image, src_prompt,None)
    if metric=="clip_similarity_target_image":
        return metrics_calculator.calculate_clip_similarity(tgt_image, tgt_prompt,None)
    if metric=="clip_similarity_target_image_edit_part":
        if tgt_mask.sum()==0:
            return "nan"
        else:
            return metrics_calculator.calculate_clip_similarity(tgt_image, tgt_prompt, tgt_mask)
    if metric == "niqe_source_image":
        return metrics_calculator.calculate_NIQE(src_save_file_niqe, img_pred_path=src_image_path, img_gt_path=None)
    if metric == "niqe_target_image":
        return metrics_calculator.calculate_NIQE(tgt_save_file_niqe, img_pred_path=None, img_gt_path=tgt_image_path)
    
def calculate_metric_video_level(metrics_calculator, metric, src_video_path, tgt_video_path, 
                                 multiple_choice_question=None, source_yes_no_question=None, target_yes_no_question=None,
                                 tgt_prompt=None, tgt_images=None, tgt_word=None, tgt_video_mask=None,
                                 ):
    if metric in {"motion_fidelity_score", "motion_fidelity_score_edit_part"}:
        return metrics_calculator.calculate_motion_fidelity_score(
            src_video_path, tgt_video_path, 
            video_masks=tgt_video_mask if metric == "motion_fidelity_score_edit_part" else None
        )
    elif metric == "five_acc":
        return metrics_calculator.calculate_five_acc(
            source_yes_no_question, target_yes_no_question, multiple_choice_question, tgt_video_path
        )
    else:
        raise ValueError(f"Metric {metric} not supported")


def _load_r28_masks(root, edit_type, video_name, frame_stride, src_size):
    """✨ R28: read a dumped mask npz as a list of ``[H, W, 3]`` masks, or None.

    Returns None when `root` is unset (the default path, where every union metric reports
    'nan') or the clip has no dump. The dumped stride MUST match the stride this run
    scores at: evaluate.py strides both sides and zips them positionally, so a mismatch
    would misalign every mask after the first rather than fail visibly.
    """
    if not root:
        return None
    path = os.path.join(root, f"edit{edit_type}", f"{video_name}.npz")
    if not os.path.exists(path):
        print(f"[R28] no target mask for edit{edit_type}/{video_name}; union metrics -> nan")
        return None
    with np.load(path, allow_pickle=False) as d:
        shape = tuple(int(v) for v in d["shape"])
        stride = int(d["stride"][0])
        if stride != frame_stride:
            raise ValueError(
                f"{path}: dumped at frame_stride {stride} but this run scores at "
                f"{frame_stride}; masks and frames would be misaligned."
            )
        arr = np.unpackbits(d["M"], axis=-1)[:, : shape[1] * shape[2]].reshape(shape).astype(bool)
    w, h = src_size
    out = []
    for m in arr:
        if m.shape != (h, w):
            m = np.array(Image.fromarray(m.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)) > 127
        out.append(m[:, :, np.newaxis].repeat(3, axis=2))
    return out


def list_images(directory):
    image_extensions = ('*.png', '*.jpg', '*.jpeg')

    # Create a list to store image paths
    image_files = []
    
    # Loop through each extension and find matching files
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(directory, ext)))
    
    return sorted(image_files)

def mp4_to_frames_ffmpeg(video_path):
    output_dir = video_path.replace(".mp4", "")
    os.makedirs(output_dir, exist_ok=True)

    # Use ffmpeg to extract frames
    output_pattern = os.path.join(output_dir, "%05d.jpg")  # Frame naming pattern
    command = [
        "ffmpeg",
        "-i", video_path,  # Input video file
        output_pattern  # Output frame pattern
    ]

    subprocess.run(command, check=True)
    return output_dir

def load_case_filter(cases_json):
    """Build a {(video_name, editing_type_id)} whitelist, or None for no filter.

    Needed because every editN_FiVE.json holds all 100 bench videos, so
    --edit_category_list alone cannot restrict a run to a named subset.
    """
    if cases_json is None:
        return None
    with open(os.path.expanduser(cases_json), "r", encoding="utf-8") as f:
        cases = json.load(f)
    return {(c["video_name"], str(c["edit_type"])) for c in cases}


# Long-format header used when --per_frame is set.
PER_FRAME_HEADER = [
    "file_id", "video_name", "editing_type_id", "method", "frame_idx", "metric", "value",
]


def calculate_mean(evaluation_result):
    if evaluation_result is None:
        return "nan"
    
    # Filter out 'nan' values
    non_nan_values = [x for x in evaluation_result if x != "nan" and not math.isnan(x)]
    
    # If all values are 'nan', return 'nan'
    if not non_nan_values:
        return "nan"
    
    # Calculate the mean of non-'nan' values
    return sum(non_nan_values) / len(non_nan_values)

    
def main(args, config, all_tgt_video_folders, tgt_layout=None):
    annotation_mapping_files = args.annotation_mapping_files
    metrics = args.metrics
    src_image_folder = args.src_image_folder
    tgt_methods = args.tgt_methods
    edit_category_list = args.edit_category_list
    evaluate_whole_table = args.evaluate_whole_table
    frame_stride = args.frame_stride
    #✨ R34: first-chunk window. Read via getattr, matching this file's own convention for
    # later-added flags (see per_frame below), so a caller that builds its own args
    # namespace without the flag keeps the pre-R34 behaviour.
    max_frames = getattr(args, "max_frames", None)
    #✨ R36: write each clip's resized frames to a node-local temp dir, deleted once that
    # clip's metrics are done, instead of a persistent {video}_resize sibling (disk quota).
    tmp_resize = getattr(args, "tmp_resize", False)
    if args.evaluate_source_video:
        tgt_video_folders = {
            "source_videos": (os.path.join(src_image_folder, "images"), "")
        }
        args.result_path = args.result_path.replace(".csv", "_source_videos.csv")
    else:
        tgt_video_folders = {}
        if evaluate_whole_table:
            for key in all_tgt_video_folders:
                if key[0] in tgt_methods:
                    tgt_video_folders[key] = all_tgt_video_folders[key]
        else:
            for key in tgt_methods:
                tgt_video_folders[key] = all_tgt_video_folders[key]
    
    result_path = args.result_path.replace(".csv", f"_frame_stride{frame_stride}.csv")
    result_path_name = result_path.split('/')[-1]
    result_dir = '/'.join(result_path.split('/')[:-1])
    Path(result_dir).mkdir(parents=True, exist_ok=True)

    # --per_frame: keep the frame-level values that calculate_mean would collapse,
    # written to a second long-format CSV. The averaged CSV below is unaffected.
    per_frame = getattr(args, "per_frame", False)
    case_filter = load_case_filter(getattr(args, "cases_json", None))
    per_frame_path = result_path.replace(".csv", "_per_frame.csv")
    if per_frame:
        with open(per_frame_path, 'w', newline="") as f:
            csv.writer(f).writerow(PER_FRAME_HEADER)

    metrics_calculator = MetricsCalculator(args.device, config=config)
    
    result_avg_files = []
    for annotation_mapping_file in tqdm(annotation_mapping_files, desc="Evaluating annotation mapping files", total=len(annotation_mapping_files)):
        print(f"evaluating {annotation_mapping_file} ...")

        annotation_mapping_file_name = annotation_mapping_file.split("/")[-1].replace(".json", "")
        result_path = os.path.join(
            result_dir, 
            "_".join([annotation_mapping_file_name, result_path_name])
        )

        with open(result_path,'w',newline="") as f:
            csv_write = csv.writer(f)
            
            csv_head = []
            for tgt_video_folder_key, _ in tgt_video_folders.items():
                for metric in metrics:
                    if metric in {"five_acc"}:
                        csv_head.append(f"{tgt_video_folder_key}|{metric}_yes_no")
                        csv_head.append(f"{tgt_video_folder_key}|{metric}_multi_choice")
                        csv_head.append(f"{tgt_video_folder_key}|{metric}_union")
                        csv_head.append(f"{tgt_video_folder_key}|{metric}_inter")
                        csv_head.append(f"{tgt_video_folder_key}|{metric}")
                    else:
                        csv_head.append(f"{tgt_video_folder_key}|{metric}")
            
            data_row = ["file_id"] + csv_head
            csv_write.writerow(data_row)

        with open(annotation_mapping_file, "r", encoding="utf-8") as f:
            annotation_file = json.load(f)

        for key, item in tqdm(enumerate(annotation_file), desc="Evaluating videos", total=len(annotation_file)):
            if str(item["editing_type_id"]) not in edit_category_list:
                continue

            video_name = item["video_name"]
            if case_filter is not None and (video_name, str(item["editing_type_id"])) not in case_filter:
                continue
            per_frame_rows = []
            save_dir = str(item["editing_type_id"]) + "_" + item["target_prompt"][:len(item["save_dir"])-2]  # item["save_dir"]
            source_prompt = item["source_prompt"].replace("[", "").replace("]", "")
            target_prompt = item["target_prompt"].replace("[", "").replace("]", "")
            # FiVE_acc
            # "multiple_choice_question": "Is the cyclist wearing a helmet? \na) Yes \nb) No",
            # "source_yes_no_question": "Is the cyclist wearing a helmet in the image?",
            # "target_yes_no_question": "Is the cyclist not wearing a helmet in the image?"
            if "multiple_choice_question" in item:
                multiple_choice_question = item["multiple_choice_question"]
                source_yes_no_question = item["source_yes_no_question"]
                target_yes_no_question = item["target_yes_no_question"]
            else:
                multiple_choice_question = None
                source_yes_no_question = None
                target_yes_no_question = None

            src_video_path = os.path.join(src_image_folder, "images", video_name)
            src_image_names = list_images(src_video_path)[::frame_stride]
            #✨ R34: truncate BEFORE `masks` is built from this list below, so the
            # positional zip(src_images, tgt_images, masks, src_image_names,
            # tgt_image_names) in the frame loop stays aligned.
            if max_frames is not None:
                src_image_names = src_image_names[:max_frames]
            if args.evaluate_source_video:
                src_image_names = src_image_names[:40//frame_stride]

            src_images = [
                Image.open(src_image_name)
                for src_image_name in src_image_names
            ]

            mask_path = os.path.join(src_image_folder, "bmasks", video_name)
            if not os.path.exists(mask_path):
                print(f"{video_name}'s mask cannot be found!! Skip ...")
                continue

            masks = []
            for src_image_name in src_image_names:
                mask = Image.open(os.path.join(mask_path, src_image_name.split('/')[-1]))

                # Convert the mask to a numpy array and ensure it's binary (0 and 1)
                # mask = mask_decode(item["mask"])
                mask = np.array(mask)  # Convert to numpy array
                mask = (mask > 0)
                mask = mask[:,:,np.newaxis].repeat([3],axis=2)
                masks.append(mask)
        
            evaluation_result = [key]
            
            for m_i, (tgt_video_folder_key, (tgt_video_folder, terminal_folder)) in enumerate(tgt_video_folders.items()):
                src_save_file_niqe = "_".join([
                    result_path.replace(".csv", ""), "niqe_src.txt"
                ])
                tgt_save_file_niqe = "_".join([
                    result_path.replace(".csv", ""), "niqe_"+tgt_video_folder_key+"_tgt.txt"
                ])

                if not args.evaluate_source_video:
                    if tgt_layout == "flat":
                        # {root}/{video_name}; --tgt_methods already points at one arm.
                        tgt_video_name = video_name
                    elif tgt_layout == "edit_video":
                        # StreamEdit runner layout: {tgt_root}/edit{T}/{video_name}/*.png
                        prefix = os.path.basename(annotation_mapping_file)[:5]  # "editT"
                        assert prefix.startswith("edit")
                        tgt_video_name = os.path.join(prefix, video_name)
                    elif tgt_video_folder_key != "6_VideoGrain":
                        tgt_video_name = os.path.join(video_name, save_dir, terminal_folder)  # terminal_folder = "image_ode" in TokenFlow
                    else:
                        prefix = annotation_mapping_file.split('/')[-1][:5]
                        assert prefix.startswith("edit")
                        tgt_video_name = os.path.join(prefix, video_name)
                    tgt_video_path = os.path.join(tgt_video_folder, tgt_video_name)
                else:
                    tgt_video_path = src_video_path
                print(f"\n\nevaluating method: {tgt_video_folder_key}")
                
                if tgt_video_path.endswith("/"):
                    tgt_video_path = tgt_video_path[:-1]
                tgt_video_path_mp4 = tgt_video_path + '.mp4'
                if os.path.exists(tgt_video_path_mp4):   
                    # NOTE: must use ffmpeg!!
                    tgt_video_path = mp4_to_frames_ffmpeg(tgt_video_path_mp4)

                tgt_image_names = list_images(tgt_video_path)
                tgt_image_names = tgt_image_names[::frame_stride]
                #✨ R34: same window on the target side, before the resize loop -- only
                # the kept frames get written to the sibling {video}_resize dir.
                if max_frames is not None:
                    tgt_image_names = tgt_image_names[:max_frames]
                tgt_images = []
                # Only niqe_target_image reads these files back (by path); every other
                # metric uses the in-memory tgt_images.
                tmp_resize_dir = tempfile.mkdtemp(prefix="five_resize_") if tmp_resize else None
                for f_i, tgt_image_name in enumerate(tgt_image_names):
                    if tgt_image_name.endswith(".jpg") or tgt_image_name.endswith(".png"):
                        tgt_image = Image.open(tgt_image_name).resize(src_images[0].size)
                        tgt_images.append(tgt_image)

                        tgt_image_name = os.path.join(
                            tmp_resize_dir if tmp_resize_dir is not None
                            else "/".join(tgt_image_name.split('/')[:-1])+"_resize",
                            os.path.basename(tgt_image_name)
                        )
                        tgt_image_names[f_i] = tgt_image_name
                        Path("/".join(tgt_image_name.split('/')[:-1])).mkdir(parents=True, exist_ok=True)
                        tgt_image.save(tgt_image_name)

                #✨ R28: per-clip target masks and the fixed cross-arm union, loaded ONCE
                # per (clip, method). They are only read here -- the grounding models are
                # never resident in this process, which is what keeps the shared evaluator
                # off the OOM path that Qwen2.5-VL + CoTracker put it on in R20/R21/R26.
                # Rows are aligned to the STRIDED frame index, the same positional
                # convention the frame loop below zips on.
                tgt_masks_r28 = _load_r28_masks(
                    getattr(args, "tgt_mask_dir", None), item["editing_type_id"], video_name,
                    frame_stride, src_images[0].size)
                fixed_masks_r28 = _load_r28_masks(
                    getattr(args, "fixed_union_dir", None), item["editing_type_id"], video_name,
                    frame_stride, src_images[0].size)

                for m_j, metric in enumerate(metrics):
                    if metric in {"niqe_source_image"} and m_i > 0:
                        continue

                    print(f"\nevaluating metric: {metric}")
                    if len(tgt_images) == 0:
                        print(f"No images are founded {tgt_video_path}! Skip ...")
                        if metric in {"five_acc"}:
                            evaluation_result += ["nan"] * 5
                        else:
                            evaluation_result.append("nan")
                        continue
                    
                    assert len(os.listdir(src_video_path)) > 0 and \
                        len(tgt_images) > 0, f"No images are founded!"

                    try:
                        if metric in {"motion_fidelity_score", "motion_fidelity_score_edit_part", "five_acc"}:
                            if args.evaluate_source_video:
                                eval_result_ = (
                                    calculate_metric_video_level(
                                        metrics_calculator, metric,
                                        src_video_path, src_video_path,
                                        multiple_choice_question=multiple_choice_question,
                                        source_yes_no_question=source_yes_no_question,
                                        target_yes_no_question=target_yes_no_question,
                                        tgt_video_mask=masks
                                    )
                                )
                            else:
                                eval_result_ = (
                                    calculate_metric_video_level(
                                        metrics_calculator, metric,
                                        src_video_path, tgt_video_path,
                                        multiple_choice_question=multiple_choice_question,
                                        source_yes_no_question=source_yes_no_question,
                                        target_yes_no_question=target_yes_no_question,
                                        tgt_video_mask=masks
                                    )
                                )
                            # Five_acc ouputs YN-acc and MC-acc 
                            if metric in {"five_acc"}:
                                if "nan" in eval_result_:
                                    evaluation_result += ["nan"] * 5
                                else:
                                    eval_result_ =  list(eval_result_)
                                    evaluation_result_five = []
                                    for eval_result_s in list(eval_result_):
                                        evaluation_result_five.append(eval_result_s)
                                    evaluation_result_five.append(int(sum(eval_result_) > 0))
                                    evaluation_result_five.append(int(sum(eval_result_) >= len(eval_result_)))
                                    evaluation_result_five.append(calculate_mean(evaluation_result_five))
                                    evaluation_result += evaluation_result_five
                            else:
                                evaluation_result.append(eval_result_)

                            if per_frame:
                                # No frame axis: MFS compares CoTracker tracklet sets,
                                # five_acc is a VLM verdict over the clip.
                                per_frame_rows.append([
                                    key, video_name, str(item["editing_type_id"]),
                                    tgt_video_folder_key, -1, metric, eval_result_,
                                ])

                        else:
                            
                            if metric in {"niqe_source_image", "niqe_target_image"}:
                                if os.path.exists(src_save_file_niqe if metric == "niqe_source_image" else tgt_save_file_niqe):
                                    os.remove(src_save_file_niqe if metric == "niqe_source_image" else tgt_save_file_niqe)

                            evaluation_result_each_frame = []
                            for f_i, (src_image, tgt_image, mask, src_image_path, tgt_image_path,) in enumerate(zip(src_images[:len(tgt_images)], tgt_images, masks, src_image_names[:len(tgt_images)], tgt_image_names)):
                                assert src_image.size[0] == tgt_image.size[0] and src_image.size[1] == tgt_image.size[1], \
                                    f"{tgt_video_folder_key}: {src_image.size} != {tgt_image.size})"
                                
                                if args.evaluate_source_video:
                                    evaluation_result_each_frame.append(
                                        calculate_metric(
                                            metrics_calculator, metric, 
                                            src_image, src_image, 
                                            mask, mask, 
                                            source_prompt, target_prompt,
                                            src_image_path, src_image_path, 
                                            src_save_file_niqe, src_save_file_niqe,
                                        )
                                    )
                                else:
                                    evaluation_result_each_frame.append(
                                        calculate_metric(
                                            metrics_calculator, metric,
                                            src_image, tgt_image,
                                            mask, mask,
                                            source_prompt, target_prompt,
                                            src_image_path, tgt_image_path,
                                            src_save_file_niqe, tgt_save_file_niqe,
                                            #✨ R28: `mask, mask` above is unchanged, so every
                                            # existing metric keeps its behaviour; the real
                                            # target mask arrives separately and is read only
                                            # by the union families.
                                            tgt_mask_real=(tgt_masks_r28[f_i]
                                                           if tgt_masks_r28 is not None
                                                           and f_i < len(tgt_masks_r28) else None),
                                            fixed_union=(fixed_masks_r28[f_i]
                                                         if fixed_masks_r28 is not None
                                                         and f_i < len(fixed_masks_r28) else None),
                                        )
                                    )
                            
                            if per_frame:
                                # frame_idx is indexed in the *unstrided* clip, so
                                # curves from different --frame_stride share an x-axis.
                                for f_i, value in enumerate(evaluation_result_each_frame):
                                    per_frame_rows.append([
                                        key, video_name, str(item["editing_type_id"]),
                                        tgt_video_folder_key, f_i * frame_stride, metric, value,
                                    ])

                            if metric in {"niqe_source_image", "niqe_target_image"}:
                                evaluation_result.append(
                                    average_niqe_from_txt(src_save_file_niqe if metric == "niqe_source_image" else tgt_save_file_niqe)
                                )
                            else:
                                evaluation_result.append(
                                    calculate_mean(evaluation_result_each_frame)
                                )

                    except Exception as e:
                        print(f"Error: {metric}: {e}")
                        # Upstream `continue`d here without appending anything, so a
                        # raised metric dropped its cell and shifted every later column
                        # LEFT -- five_acc silently inherited motion_fidelity's value.
                        # Placeholder is the arity-aware one upstream already uses for
                        # its own no-images case (the `len(tgt_images) == 0` branch
                        # above): five_acc occupies 5 cells, every other metric 1.
                        # Triggered in practice by motion_fidelity_score_edit_part's
                        # empty-edit-mask degeneracy (R2 outcome; R14 owns the fix) --
                        # ~1 clip per 53, so ~8 rows over the full bench.
                        if metric in {"five_acc"}:
                            evaluation_result += ["nan"] * 5
                        else:
                            evaluation_result.append("nan")
                        continue

                if tmp_resize_dir is not None:
                    shutil.rmtree(tmp_resize_dir, ignore_errors=True)

            with open(result_path, 'a+', newline="") as f:
                csv_write = csv.writer(f)
                csv_write.writerow(evaluation_result)

            if per_frame and per_frame_rows:
                with open(per_frame_path, 'a', newline="") as f:
                    csv.writer(f).writerows(per_frame_rows)
 
        # calculate the average of each metric (each column)
        with open(result_path, 'r') as f:
            reader = list(csv.reader(f))
            header, rows = reader[0], reader[1:]

        avg_row = []
        # Process each column by index to handle rows with different lengths
        for col_idx, name in enumerate(header):
            print("processing", name)
            # Extract column values, handling missing values
            col_values = []
            for row in rows:
                if col_idx < len(row):
                    col_values.append(row[col_idx])
                else:
                    col_values.append("")  # Use empty string for missing values
            
            try:
                # Filter out empty strings and convert to float
                values = [float(x) for x in col_values if x != "" and x != "nan"]
                if values:  # Only calculate average if there are valid values
                    avg = sum(values) / len(values)
                    if 'structure_distance' in name:
                        avg *= 1000
                    elif 'lpips_' in name:
                        avg *= 1000
                    elif 'mse_' in name:
                        avg *= 10000
                    elif 'ssim_' in name:
                        avg *= 100
                    elif 'motion_fidelity_score' in name:
                        avg *= 100
                    elif name.startswith('five_acc'):
                        avg *= 100
                    avg_row.append(f"{avg:.4f}")
                else:
                    avg_row.append("N/A")
            except ValueError:
                avg_row.append("N/A")

        result_avg_files.append(result_path.replace('.csv', '_avg.csv'))
        with open(result_avg_files[-1], 'w', newline='') as f_out:
            writer = csv.writer(f_out)
            writer.writerow(header)
            writer.writerow(avg_row)
    
    # average the results in result_avg_files
    if result_avg_files:
        all_avg_rows = []
        
        # Read all average files
        for result_avg_file in result_avg_files:
            with open(result_avg_file, 'r') as f:
                reader = list(csv.reader(f))
                header, rows = reader[0], reader[1:]
                if rows:  # Make sure there's data
                    all_avg_rows.append(rows[0])  # Get the average row
        
        # Calculate final averages across all files
        final_avg_row = []
        for col_idx, name in enumerate(header):
            print("final averaging", name)
            
            # Extract values from all average files for this column
            col_values = []
            for avg_row in all_avg_rows:
                if col_idx < len(avg_row) and avg_row[col_idx] != "N/A":
                    try:
                        col_values.append(float(avg_row[col_idx]))
                    except ValueError:
                        pass  # Skip non-numeric values
            
            # Calculate final average
            if col_values:
                final_avg = sum(col_values) / len(col_values)
                final_avg_row.append(f"{final_avg:.4f}")
            else:
                final_avg_row.append("N/A")
        
        # Write final averaged results
        with open(f"{os.path.dirname(result_avg_files[0])}/final_averaged_results.csv", 'w', newline='') as f_out:
            writer = csv.writer(f_out)
            writer.writerow(header)
            writer.writerow(final_avg_row)

        # Also copy to a deterministic, result_path-derived name so downstream tasks (e.g. R3)
        # can find a stable overall CSV (final_averaged_results.csv is overwritten each run).
        avg_name = os.path.basename(args.result_path).replace(".csv", "") + "_avg.csv"
        avg_out = os.path.join(os.path.dirname(result_avg_files[0]), avg_name)
        with open(avg_out, 'w', newline='') as f_out:
            writer = csv.writer(f_out)
            writer.writerow(header)
            writer.writerow(final_avg_row)
        print(f"[evaluate] overall metrics -> {avg_out}")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_stride", type=int, default=8)
    #✨ R34: keep only the first N frames AFTER striding, on BOTH sides. Default None
    # leaves every existing call site bit-identical. R34 scores the first rollout chunk
    # (num_frame_per_block=3 latent frames = pixel frames 0-8): --frame_stride 1
    # --max_frames 9.
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Keep only the first N frames after striding (source and "
                             "target). Default: no truncation.")
    parser.add_argument("--tmp_resize", action="store_true",
                        help="Write resized target frames to a per-clip temp dir ($TMPDIR) "
                             "deleted after the clip is scored, instead of a persistent "
                             "{video}_resize sibling. Metric values are unchanged.")
    parser.add_argument('--annotation_mapping_files', nargs = '+', type=str, default=[
                                                        "data/edit_prompt/edit1_FiVE.json",
                                                        "data/edit_prompt/edit2_FiVE.json",
                                                        "data/edit_prompt/edit3_FiVE.json",
                                                        "data/edit_prompt/edit4_FiVE.json",
                                                        "data/edit_prompt/edit5_FiVE.json",
                                                        "data/edit_prompt/edit6_FiVE.json",
                                                        ])
    parser.add_argument('--metrics', nargs = '+', type=str, default=[
                                                         "structure_distance",
                                                         "psnr_unedit_part",
                                                         "lpips_unedit_part",
                                                         "mse_unedit_part",
                                                         "ssim_unedit_part",
                                                         "clip_similarity_source_image",
                                                         "clip_similarity_target_image",
                                                         "clip_similarity_target_image_edit_part",
                                                        #  "niqe_source_image",
                                                         "niqe_target_image",
                                                         "motion_fidelity_score",
                                                         "motion_fidelity_score_edit_part",
                                                         "five_acc",    
                                                        ])
    parser.add_argument('--src_image_folder', type=str, default="data/")
    parser.add_argument('--tgt_methods', nargs = '+', type=str, default=[
                                                                    # "1_TokenFlow",
                                                                    # "2_DMT",
                                                                    # "4_VidToMe",
                                                                    # "5_AnyV2V",
                                                                    # "6_VideoGrain",
                                                                    # "7_Pyramid_Edit",
                                                                    "8_Wan_Edit",
                                                                  ])
    parser.add_argument('--result_path', type=str, default="outputs/evaluation_result.csv")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--edit_category_list',  nargs = '+', type=str, default=[
                                                                                "1",
                                                                                "2",
                                                                                "3",
                                                                                "4",
                                                                                "5",
                                                                                "6",
                                                                                ]) # the editing category that needed to run
    parser.add_argument('--evaluate_whole_table', action= "store_true") # rerun existing images
    parser.add_argument('--evaluate_source_video', action= "store_true")
    parser.add_argument('--config_path', type=str, default="config.yaml")
    # StreamEdit additions: choose target-folder layout + CSV column key when --tgt_methods is a path root
    parser.add_argument('--tgt_layout', type=str, default=None, choices=["edit_video", "fivebench", "flat"],
                        help="edit_video={root}/edit{T}/{video_name} (R1/R3); "
                             "fivebench={video_name}/{save_dir}/{terminal} (native FiVE-Bench); "
                             "flat={root}/{video_name} (R10: one arm per invocation)")
    parser.add_argument('--per_frame', action="store_true",
                        help="additionally emit a long-format per-frame CSV "
                             "(file_id,video_name,editing_type_id,method,frame_idx,metric,value) "
                             "alongside the averaged one; whole-clip metrics get frame_idx=-1. "
                             "Default off => averaged output is bit-identical to before.")
    parser.add_argument('--cases_json', type=str, default=None,
                        help="restrict evaluation to the (video_name, edit_type) pairs listed "
                             "in this cases.json (editN_FiVE.json each hold all 100 videos)")
    #✨ R28: both default to None => every existing metric is byte-identical and the
    # union families report 'nan'. Neither loads a model in this process.
    parser.add_argument('--tgt_mask_dir', type=str, default=None,
                        help="R28: root of this ARM's dumped target masks, "
                             "{root}/edit{T}/{video}.npz (see evaluation/r28_target_masks.py). "
                             "Enables the *_unedit_union metrics")
    parser.add_argument('--fixed_union_dir', type=str, default=None,
                        help="R28: root of the FIXED cross-arm union, same layout "
                             "(see evaluation/r28_fixed_union.py). Enables the "
                             "*_unedit_union_fixed metrics, which score every arm on an "
                             "identical background region")
    parser.add_argument('--tgt_key', type=str, default=None,
                        help="CSV column key for the method (default: basename of the --tgt_methods path)")
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    args_dict = vars(args)
    for key, value in args_dict.items():
        if key in config and value is not None:
            config[key] = value
    
    # NOTE: Modify the target video folders here!!!!! 
    all_tgt_video_folders = {
        # "1_TokenFlow": (f"{config.root_tgt_video_folder}/TokenFlow/", "img_ode"),
        # "2_DMT": (f"{config.root_tgt_video_folder}/diffusion-motion-transfer/", "result_frames"),
        # "4_VidToMe": (f"{config.root_tgt_video_folder}/VidToMe/", "frames"),
        # "5_AnyV2V": (f"{config.root_tgt_video_folder}/AnyV2V/Results/Prompt-Based-Editing_frames32/i2vgen-xl", "ddim_init_latents_t_idx_0_nsteps_50_cfg_9.0_pnpf0.2_pnps0.2_pnpt0.5"),
        # "6_VideoGrain": (f"{config.root_tgt_video_folder}/video_grain/", ""),
        "7_Pyramid_Edit": (f"{config.root_tgt_video_folder}/Pyramid-edit/", "result_all_frames"),
        "8_Wan_Edit": (f"{config.root_tgt_video_folder}/Wan-Edit/", ""),
        "StreamGVE_SF": (f"{config.root_tgt_video_folder}/StreamGVE_SF/", ""),
    }

    # StreamEdit: when --tgt_layout is set, --tgt_methods is a single method-root PATH
    # (e.g. /projects/dataggen/outputs/five_bench/baseline); build the folder map directly.
    if args.tgt_layout is not None:
        tgt_root = args.tgt_methods[0].rstrip('/')
        tgt_key = args.tgt_key or os.path.basename(tgt_root)
        all_tgt_video_folders = {tgt_key: (tgt_root, "")}
        args.tgt_methods = [tgt_key]

    main(args, config, all_tgt_video_folders, tgt_layout=args.tgt_layout)