import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(key="example", video_processor_factory=VideoProcessor)