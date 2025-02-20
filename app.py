import streamlit as st
import os
import tempfile
import subprocess
import cv2
import torch
from main import process_video

def handle_file_upload(uploaded_file):
    """Handles file upload and returns a temporary file path."""
    if uploaded_file:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_file.read())
        temp_file.close()
        return temp_file.name
    return None

def convert_video_with_ffmpeg(input_video_path, output_video_path):
    """Converts video using FFmpeg for better compatibility."""
    command = f'ffmpeg -loglevel error -i "{input_video_path}" -vcodec libx264 -crf 23 -preset fast "{output_video_path}"'
    result = subprocess.run(command, shell=True)

    if result.returncode != 0:
        st.error(f'FFmpeg encountered an error while processing the video. ❌')
        st.error(f'Temporary input video file retained at: {input_video_path}')
        st.error(f'Temporary output video file retained at: {output_video_path}')
    else:
        os.remove(input_video_path)  # Remove temp input file if successful
        st.success(f'Video processed successfully and saved at: {output_video_path}')

def main():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    st.title("Anomaly Video Synopsis")

    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    text_input = st.text_input("Enter text input for processing", value="person carrying a weapon")
    
    use_flow = st.toggle("Use Optical Flow (RAFT) instead of Background Subtraction", value=False)
    
    raft_path = None
    if use_flow:
        raft_path = st.text_input("Path to RAFT directory", value="/kaggle/input/raft-pytorch")
    
    if uploaded_video:
        input_video_path = handle_file_upload(uploaded_video)
        output_video_path = tempfile.mktemp(suffix=".mp4")
        mask_video_path = tempfile.mktemp(suffix=".mp4")

        # Video metadata
        video_cap = cv2.VideoCapture(input_video_path)
        if video_cap.isOpened():
            frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_length = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(video_cap.get(cv2.CAP_PROP_FPS))

            st.write(f"📽️ **Video Details:** {frame_width}x{frame_height}, {fps} FPS, {video_length} frames")
        else:
            st.error("Failed to open video for metadata extraction.")
        video_cap.release()

        if st.button("Process Video"):
            if use_flow and not raft_path:
                st.error("⚠️ RAFT path is required when using optical flow!")
                return

            st.info("Processing video, please wait...")

            # Call process_video with the selected method
            process_video(
                input_video_path=input_video_path,
                output_video_path=output_video_path,
                mask_video_path=mask_video_path,
                text_input=text_input,
                device=DEVICE,
                use_flow=use_flow,
                raft_path=raft_path
            )

            st.success("Processing completed successfully! ✅")

            # Convert video with FFmpeg for better playback
            final_video_name = output_video_path.replace(".mp4", "_final.mp4")
            convert_video_with_ffmpeg(output_video_path, final_video_name)

            # Display videos
            if os.path.exists(final_video_name):
                st.video(final_video_name, format='video/mp4', start_time=0)
            else:
                st.error("Failed to process video correctly with FFmpeg. ❌")

if __name__ == "__main__":
    main()
