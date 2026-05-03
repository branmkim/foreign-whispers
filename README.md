# Foreign Whispers

Brandon Kim, bmk7319@nyu.edu

[Link to sample output (Google Drive)](https://drive.google.com/file/d/12xoGLLQV-Fk56nBoRY_jZ_dLratk_Nia/view?usp=sharing)

I did not have access to an NVIDIA GPU for this assignment, so I ran the Chatterbox TTS API on a RunPod instance instead. I did not need the Whisper STT container because I used the YouTube captions, and the frontend and API containers were run on my local Macbook without any issue. There are some different configurations I have set in my project to be able to interact with my RunPod instance instead of a local TTS container. While working on it, I encountered some issues in connecting everything together and working around rate limiting and errors on RunPod, but the final product is comparable to those from locally-run TTS.

[Link to old README](./README_old.md)