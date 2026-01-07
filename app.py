import streamlit as st
from audiocraft.models import MusicGen

@st.cache_resource
def load_model():
    # 'small' মডেলটি ব্যবহার করুন কারণ এটি সবচেয়ে কম মেমোরি নেয়
    model = MusicGen.get_pretrained('facebook/musicgen-small')
    return model

# অ্যাপের বাকি অংশ...
from audiocraft.models import MusicGen
import torch
import torchaudio
import os
import subprocess
import sys

# নিশ্চিত করা যে সব ডিপেন্ডেন্সি ঠিক আছে
def install_dependencies():
    try:
        import audiocraft
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "audiocraft"])

# পেজ সেটআপ এবং ডিজাইন
st.set_page_config(page_title="AI Melody Maker", page_icon="🎵", layout="centered")

# কাস্টম CSS দিয়ে UI সুন্দর করা
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #ff3333;
        border: none;
        transform: scale(1.02);
    }
    .title-text {
        text-align: center;
        color: #FF4B4B;
        font-family: 'Helvetica', sans-serif;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        text-align: center;
        color: #888;
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# হেডার সেকশন
st.markdown("<h1 class='title-text'>🎵 AI Melody Maker</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #bbb;'>আপনার কল্পনাকে শব্দে রূপান্তর করুন</p>", unsafe_allow_html=True)
st.write("---")

# মডেল লোডিং ফাংশন
@st.cache_resource
def load_model():
    return MusicGen.get_pretrained('facebook/musicgen-small')

# সাইডবার বা মেনু
with st.sidebar:
    st.header("সেটিংস")
    duration = st.slider("গানের দৈর্ঘ্য (সেকেন্ড)", 5, 20, 10)
    st.info("দ্রষ্টব্য: ছোট মডেল দ্রুত কাজ করে।")

# মেইন ইন্টারফেস
col1, col2 = st.columns([2, 1])

with col1:
    prompt = st.text_area("আপনার গানটি কেমন হবে বর্ণনা করুন:", 
                          placeholder="উদাহরণ: Lofi hip hop beat with smooth saxophone and rain sounds...",
                          height=100)

with col2:
    st.write("সহায়তা:")
    st.caption("১. বাদ্যযন্ত্রের নাম লিখুন।")
    st.caption("২. গানের মুড (Happy, Sad) লিখুন।")

# জেনারেট বাটন
if st.button("Generate Magic Music ✨"):
    if prompt:
        try:
            with st.status("AI সুর তৈরি করছে...", expanded=True) as status:
                model = load_model()
                model.set_generation_params(duration=duration)
                
                wav = model.generate([prompt])
                
                file_path = "generated_music.wav"
                torchaudio.save(file_path, wav[0].cpu(), 32000)
                status.update(label="সুর তৈরি সম্পন্ন!", state="complete", expanded=False)

            # অডিও ডিসপ্লে
            st.audio(file_path, format="audio/wav")
            
            # ডাউনলোড বাটন
            with open(file_path, "rb") as f:
                st.download_button(
                    label="📥 ডাউনলোড করুন",
                    data=f,
                    file_name="ai_music.wav",
                    mime="audio/wav"
                )
            st.balloons()
            
        except Exception as e:
            st.error(f"দুঃখিত, একটি সমস্যা হয়েছে: {e}")
    else:
        st.warning("আগে কিছু লিখুন!")

# ফুটার
st.markdown("<div class='footer'>Made with ❤️ by AI Melody Maker Team</div>", unsafe_allow_html=True)
