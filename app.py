import os, json, re, asyncio, tempfile, requests
import numpy as np
import streamlit as st
import edge_tts
from gtts import gTTS
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import (VideoFileClip, AudioFileClip, ImageClip, ColorClip,
                            CompositeVideoClip, concatenate_videoclips, vfx)

MODEL = "claude-sonnet-5-5"
W, H = 1080, 1920
VOICES = {"English (Aria, female)": "en-US-AriaNeural", "English (Guy, male)": "en-US-GuyNeural",
          "Tamil (Pallavi, female)": "ta-IN-PallaviNeural", "Tamil (Valluvar, male)": "ta-IN-ValluvarNeural",
          "Hindi (Swara, female)": "hi-IN-SwaraNeural"}
NICHES = ["Amazing facts", "Money tips", "History mysteries", "Motivation", "Scary stories", "AI and tech news"]
FONTS = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "DejaVuSans-Bold.ttf"]
TAMIL_FONTS = ["/usr/share/fonts/truetype/noto/NotoSansTamil-Bold.ttf",
               "/usr/share/fonts/truetype/noto/NotoSansTamil-Regular.ttf", "NotoSansTamil-Bold.ttf"]


def secret(name):
    try:
        return st.secrets[name]
    except Exception:
        return os.getenv(name, "")


def get_script(niche, topic, n, voice, key):
    import anthropic
    lang = "Tamil" if voice.startswith("ta") else "Hindi" if voice.startswith("hi") else "English"
    prompt = (f"Create a YouTube Short in the niche '{niche}'. Topic: {topic or 'pick a fresh, surprising topic'}. "
              f"Narration language: {lang}. Exactly {n} scenes. Scene 1 is a strong hook. Each scene: 1-2 short "
              "sentences of narration and 2-3 English stock-footage search keywords. Reply ONLY with JSON: "
              '{"title": "...", "description": "... with hashtags", "scenes": [{"narration": "...", "keywords": "..."}]}')
    msg = anthropic.Anthropic(api_key=key).messages.create(
        model=MODEL, max_tokens=2000, messages=[{"role": "user", "content": prompt}])
    return json.loads(re.search(r"\{.*\}", msg.content[0].text, re.S).group(0))


def parse_manual(text):
    scenes = []
    for line in text.strip().splitlines():
        if line.strip():
            narr, _, kw = line.partition("|")
            scenes.append({"narration": narr.strip(), "keywords": kw.strip() or "nature landscape"})
    return {"title": "", "description": "", "scenes": scenes}


def tts(text, voice, path):
    try:
        asyncio.run(edge_tts.Communicate(text, voice).save(path))
        if os.path.getsize(path) < 1000:
            raise RuntimeError("empty audio")
    except Exception:
        gTTS(text, lang=voice[:2]).save(path)


def fetch_clip(query, key, folder, used):
    try:
        r = requests.get("https://api.pexels.com/videos/search", headers={"Authorization": key},
                         params={"query": query, "orientation": "portrait", "per_page": 10}, timeout=20).json()
        for v in r.get("videos", []):
            if v["id"] in used:
                continue
            files = sorted([f for f in v["video_files"] if f.get("height", 0) >= 1280], key=lambda f: f["height"])
            if files:
                used.add(v["id"])
                path = os.path.join(folder, f"{v['id']}.mp4")
                with open(path, "wb") as fh:
                    fh.write(requests.get(files[0]["link"], timeout=90).content)
                return path
    except Exception:
        pass
    return None


def load_font(size, text=""):
    fonts = (TAMIL_FONTS if re.search(r"[\u0B80-\u0BFF]", text) else []) + FONTS
    try:
        from PIL import features
        eng = ImageFont.Layout.RAQM if features.check("raqm") else ImageFont.Layout.BASIC
    except Exception:
        eng = None
    for f in fonts:
        for kw in ({"layout_engine": eng} if eng is not None else {}, {}):
            try:
                return ImageFont.truetype(f, size, **kw)
            except Exception:
                continue
    return ImageFont.load_default()


def caption(text, dur):
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d, font = ImageDraw.Draw(img), load_font(58, text)
    lines, line = [], ""
    for w in text.split():
        t = (line + " " + w).strip()
        if d.textlength(t, font=font) > W - 160:
            lines.append(line)
            line = w
        else:
            line = t
    lines.append(line)
    y = 1400 - len(lines) * 40
    for l in lines:
        d.text(((W - d.textlength(l, font=font)) / 2, y), l, font=font, fill="white", stroke_width=5, stroke_fill="black")
        y += 80
    return ImageClip(np.array(img)).set_duration(dur)


def build(scenes, pexels_key, voice, captions, out, log):
    tmp, clips, used, missed = tempfile.mkdtemp(), [], set(), 0
    for i, s in enumerate(scenes):
        log(f"Scene {i + 1}/{len(scenes)}: voice and footage...")
        mp3 = os.path.join(tmp, f"v{i}.mp3")
        tts(s["narration"], voice, mp3)
        audio = AudioFileClip(mp3)
        dur = audio.duration + 0.3
        q = s["keywords"] if s["keywords"].isascii() else "nature landscape"
        path = fetch_clip(q, pexels_key, tmp, used) if pexels_key else None
        if path:
            v = VideoFileClip(path).without_audio()
            v = v.fx(vfx.loop, duration=dur) if v.duration < dur else v.subclip(0, dur)
            v = v.resize(max(W / v.w, H / v.h))
            v = v.crop(x_center=v.w / 2, y_center=v.h / 2, width=W, height=H)
        else:
            missed += 1
            v = ColorClip((W, H), color=(20, 20, 40)).set_duration(dur)
        layers = [v] + ([caption(s["narration"], dur)] if captions else [])
        clips.append(CompositeVideoClip(layers, size=(W, H)).set_audio(audio))
    log("Rendering final video (1 to 3 minutes)...")
    concatenate_videoclips(clips, method="compose").write_videofile(
        out, fps=24, codec="libx264", audio_codec="aac", preset="ultrafast", threads=2, logger=None)
    return missed


st.set_page_config(page_title="AutoShort AI", page_icon="🎬", layout="centered")
st.title("🎬 AutoShort AI")
claude_key, pexels_key = secret("ANTHROPIC_API_KEY"), secret("PEXELS_API_KEY")
if not pexels_key:
    st.warning("PEXELS_API_KEY is missing, so videos will have a plain dark background. Add it in app Settings > Secrets.")

niche = st.selectbox("Niche", NICHES)
topic = st.text_input("Topic (optional)", placeholder="Leave empty and AI picks one")
voice_name = st.selectbox("Voice", list(VOICES))
c1, c2 = st.columns(2)
n = c1.slider("Scenes", 3, 8, 5)
caps = c2.checkbox("Captions", True)
manual = st.text_area("Or paste your own script", placeholder="One scene per line: narration | keywords")

if st.button("Generate video", type="primary", use_container_width=True):
    try:
        with st.status("Working...", expanded=True) as status:
            if manual.strip():
                data = parse_manual(manual)
            elif claude_key:
                status.write("Writing script...")
                data = get_script(niche, topic, n, VOICES[voice_name], claude_key)
            else:
                st.error("No ANTHROPIC_API_KEY found. Paste your own script instead.")
                st.stop()
            out = os.path.join(tempfile.mkdtemp(), "short.mp4")
            missed = build(data["scenes"], pexels_key, VOICES[voice_name], caps, out, status.write)
            status.update(label="Done", state="complete")
        st.session_state.update(video=open(out, "rb").read(), data=data, missed=missed)
    except Exception as e:
        st.error("Something failed. Send this error to Claude:")
        st.exception(e)

if "video" in st.session_state:
    d = st.session_state["data"]
    if st.session_state.get("missed"):
        st.info(f"{st.session_state['missed']} scene(s) had no stock clip and use a plain background.")
    st.video(st.session_state["video"])
    st.download_button("Download MP4", st.session_state["video"], "short.mp4", "video/mp4", use_container_width=True)
    if d.get("title"):
        st.subheader("Title")
        st.code(d["title"], language=None)
        st.subheader("Description")
        st.code(d["description"], language=None)
