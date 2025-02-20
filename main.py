import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import os
import argparse

from utils.utils import annotate_image, MASK_ANNOTATOR
from utils.florence import florence_load_model, florence_run_inference, TASK_CAPTION_TO_PHRASE_GROUNDING
from utils.sam import initialize_sam, perform_sam_inference
import supervision as sv

from video_flow import load_raft_model, compute_flow_and_foreground

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

FLORENCE_MODEL, FLORENCE_PROCESSOR = florence_load_model(device=DEVICE)
SAM_MODEL = initialize_sam(device=DEVICE)

def run_model_inference(image: Image.Image, text_input: str):
    """Runs Florence model inference on the input image."""
    _, response = florence_run_inference(
        model=FLORENCE_MODEL,
        processor=FLORENCE_PROCESSOR,
        device=DEVICE,
        image=image,
        task=TASK_CAPTION_TO_PHRASE_GROUNDING,
        text=text_input
    )
    return response.get(TASK_CAPTION_TO_PHRASE_GROUNDING, {}).get('bboxes', []), response

def process_frame(frame, background_subtractor, text_input, large_detection_threshold):
    """Processes a single frame using background subtraction & Florence model."""
    fg_mask = background_subtractor.apply(frame)
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    significant_motion = any(cv2.contourArea(contour) > frame.size // 90 for contour in contours)

    if not significant_motion:
        return frame, frame  

    image_input = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    bbox_data, response = run_model_inference(image_input, text_input)

    for bbox in bbox_data:
        bbox_area = bbox[2] * bbox[3]  
        if bbox_area > large_detection_threshold:
            continue  

        detections = sv.Detections.from_lmm(
            lmm=sv.LMM.FLORENCE_2,
            result=response,
            resolution_wh=image_input.size
        )
        detections, score = perform_sam_inference(SAM_MODEL, image_input, detections)
        annotated_image = annotate_image(image_input, detections)
        mask_image = MASK_ANNOTATOR.annotate(image_input.copy(), detections)

        output_frame = cv2.cvtColor(np.array(annotated_image), cv2.COLOR_RGB2BGR)
        mask_frame = cv2.cvtColor(np.array(mask_image), cv2.COLOR_RGB2BGR)
        return output_frame, mask_frame

    return frame, frame 

def process_video(input_video_path, output_video_path, mask_video_path, text_input, use_flow=False, raft_path=None, **kwargs):
    """Processes a video using either traditional background subtraction or RAFT optical flow."""
    if not os.path.exists(input_video_path):
        print("Error: The file does not exist.")
        return

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Failed to open video.")
        return

    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
    frame_area = frame_width * frame_height
    large_detection_threshold = frame_area * 0.75

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, kwargs['fps'], (frame_width, frame_height))
    mask_out = cv2.VideoWriter(mask_video_path, fourcc, kwargs['fps'], (frame_width, frame_height))

    if use_flow:
        raft_model, InputPadder = load_raft_model(DEVICE, raft_path)

    background_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=kwargs['history'], varThreshold=kwargs['var_threshold'], detectShadows=kwargs['detect_shadows']
    )

    with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Processing Video", unit="frames") as pbar:
        ret, prev_frame = cap.read()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if use_flow:
                prev_tensor = torch.from_numpy(prev_frame).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
                curr_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)

                _, mask, foreground = compute_flow_and_foreground(prev_tensor, curr_tensor, raft_model, InputPadder, DEVICE)

                output_frame = foreground
                mask_frame = mask
                prev_frame = frame  
            else:
                output_frame, mask_frame = process_frame(frame, background_subtractor, text_input, large_detection_threshold)

            out.write(output_frame)
            mask_out.write(mask_frame)

            pbar.update(1)

    cap.release()
    out.release()
    mask_out.release()

def main():
    parser = argparse.ArgumentParser(description='Process video for specific text input.')
    parser.add_argument('--input_video_path', type=str, default='vid_src/6_new.mp4', help='Path to input video')
    parser.add_argument('--output_video_path', type=str, default='output.mp4', help='Path to save output')
    parser.add_argument('--mask_video_path', type=str, default='mask_output.mp4', help='Path to save mask output')
    parser.add_argument('--text_input', type=str, default='person carrying a weapon', help='Text input for processing')
    parser.add_argument('--fps', type=int, default=20, help='Frames per second')
    parser.add_argument('--history', type=int, default=500, help='BG subtraction history length')
    parser.add_argument('--var_threshold', type=int, default=16, help='BG subtraction threshold')
    parser.add_argument('--detect_shadows', type=bool, default=True, help='Enable shadow detection')
    parser.add_argument('--use_flow', action='store_true', default=False, help='Use optical flow (RAFT) instead of background subtraction')
    parser.add_argument('--raft_path', type=str, default='/kaggle/input/raft-pytorch', help='Path to RAFT directory (required if --use_flow)')

    args = parser.parse_args()

    process_video(
        args.input_video_path,
        args.output_video_path,
        args.mask_video_path,
        args.text_input,
        use_flow=args.use_flow,
        raft_path=args.raft_path,
        fps=args.fps,
        history=args.history,
        var_threshold=args.var_threshold,
        detect_shadows=args.detect_shadows
    )

if __name__ == "__main__":
    main()
