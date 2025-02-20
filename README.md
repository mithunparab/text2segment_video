# Simple Video Summarization using Text-to-Segment Anything (Florence2 + SAM2)

This project provides a video processing tool that utilizes advanced AI models, specifically Florence2 and SAM2, to detect and segment specific objects or activities in a video based on textual descriptions. The system identifies significant motion in video frames and then performs deep learning inference to locate objects or actions described by the user's textual input.

<div style="display: flex; align-items: center; gap: 10px;">
    <a href="https://www.kaggle.com/code/mithunparab/simple-video-summarization-using-text-to-segment-a" target="_blank">
        <img src="https://kaggle.com/static/images/site-logo.png" alt="Kaggle Notebook" height="40" style="margin-bottom: -15px;">
    </a>
    <a href="https://colab.research.google.com/drive/1PcK_6anMRYnRcmOw5TkwFUT8lrXYoHUW?usp=sharing" target="_blank">
        <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
    </a>
</div>

---

## Installation

Before running the script, ensure that all dependencies are installed. You can install the necessary packages using the following command:

```bash
pip install -r requirements.txt
```

For downloading model checkpoints, run the following commands:

```bash
cd checkpoints
./download_ckpts.sh
cd ..
```

## Requirements

- Python 3.7+
- OpenCV
- Pillow (PIL)
- PyTorch
- tqdm

Additionally, install the following packages:

```bash
pip install -q einops spaces timm transformers samv2 gradio supervision opencv-python
```

## Usage

The video processing can be executed from the command line with various arguments to specify the input video, output video, mask video, text input, and processing options.

### Basic Command

```bash
python main.py --input_video_path <path_to_input_video> --output_video_path <path_to_output_video> --mask_video_path <path_to_mask_video> --text_input "your text here"
```

### Parameters

- `--input_video_path`  
  **Required.** Path to the source video file.

- `--output_video_path`  
  **Required.** Path to save the processed output video.

- `--mask_video_path`  
  **Required.** Path to save the mask video that highlights detected objects.

- `--text_input`  
  **Required.** Textual description of the object or activity to detect and segment in the video.

- `--fps`  
  Frames per second for the output video. Default is 20.

- `--history`  
  Background subtraction history length. Default is 500.

- `--var_threshold`  
  Background subtraction threshold. Default is 16.

- `--detect_shadows`  
  Enable shadow detection in background subtraction. Default is True.

- `--use_flow`  
  Toggle to use RAFT-based optical flow instead of background subtraction. Default is False.

- `--raft_path`  
  Path to the RAFT directory (required if `--use_flow` is enabled). Default is `/kaggle/input/raft-pytorch`.

### Example Command (Using Background Subtraction)

```bash
python main.py --input_video_path ./input_video.mp4 --output_video_path ./output_video.mp4 --mask_video_path ./mask_video.mp4 --text_input "person carrying a weapon"
```

### Example Command (Using RAFT Optical Flow)

```bash
python main.py --input_video_path ./input_video.mp4 --output_video_path ./output_video.mp4 --mask_video_path ./mask_video.mp4 --text_input "person carrying a weapon" --use_flow --raft_path /path/to/raft
```

## Web Interface

A web-based user interface is available using Streamlit. To launch the web interface, run:

```bash
streamlit run app.py
```

## Features

- **Motion Detection:**  
  Detect significant motion in the video to focus processing on relevant segments.

- **Object and Action Detection:**  
  Utilize state-of-the-art models (Florence2 and SAM2) to detect and segment objects or actions based on the provided text input.

- **Dual Processing Modes:**  
  Choose between traditional background subtraction and RAFT-based optical flow for foreground extraction.

- **Output Generation:**  
  Generate an annotated video along with a corresponding mask video showing the detected segments.

## Notes

1. When using optical flow, ensure that the RAFT model and its weights are correctly set up in your environment.
2. The web interface (app.py) allows you to upload videos and toggle between processing modes, providing a convenient user experience.

## To Do

- [x] **WebUI**
- [ ] **Robust Video Synopsis**

## Related work

- <https://github.com/facebookresearch/segment-anything-2/tree/main>
- <https://huggingface.co/spaces/SkalskiP/florence-sam>
- <https://huggingface.co/collections/microsoft/florence-6669f44df0d87d9c3bfb76de>
