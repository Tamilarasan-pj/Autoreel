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
QUALITY = {"480p (fast)": (480, 854), "720p (HD)": (720, 1280), "1080p (Full HD)": (1080, 1920)}
BITRATE = {480: "2000k", 720: "4500k", 1080: "8000k"}
VOICES = {"English (Aria, female)": "en-US-AriaNeural", "English (Guy, male)": "en-US-GuyNeural",
          "Tamil (Pallavi, female)": "ta-IN-PallaviNeural", "Tamil (Valluvar, male)": "ta-IN-ValluvarNeural",
          "Hindi (Swara, female)": "hi-IN-SwaraNeural"}
NICHES = ["Amazing facts", "Money tips", "History mysteries", "Motivation", "Scary stories", "AI and tech news"]
FONTS = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "DejaVuSans-Bold.ttf"]
TAMIL_FONTS = ["/usr/share/fonts/truetype/noto/NotoSansTamil-Bold.ttf",
               "/usr/share/fonts/truetype/noto/NotoSansTamil-Regular.ttf", "NotoSansTamil-Bold.ttf"]


@st.cache_data(ttl=86400)
def load_voices():
    try:
        out = {}
        for v in sorted(asyncio.run(edge_tts.list_voices()), key=lambda v: (v["Locale"], v["ShortName"])):
            if v["Locale"][:2] in ("ta", "en", "hi"):
                name = v["ShortName"].split("-", 2)[2].replace("Neural", "")
                out[f'{name} | {v["Gender"]} | {v["Locale"]}'] = v["ShortName"]
        return out or VOICES
    except Exception:
        return VOICES


def secret(name):
    try:
        return st.secrets[name]
    except Exception:
        return os.getenv(name, "")


GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-flash-latest"]


def script_prompt(niche, topic, n, voice):
    lang = "Tamil" if voice.startswith("ta") else "Hindi" if voice.startswith("hi") else "English"
    return (f"Create a YouTube Short in the niche '{niche}'. Topic: {topic or 'pick a fresh, surprising topic'}. "
            f"Narration language: {lang}. Exactly {n} scenes. Scene 1 is a strong hook. Each scene: 1-2 short "
            "sentences of narration and 2-3 English stock-footage search keywords (visual things like 'ocean waves'). "
            'Reply ONLY with JSON: {"title": "...", "description": "... with hashtags", '
            '"scenes": [{"narration": "...", "keywords": "..."}]}')


def get_script_gemini(niche, topic, n, voice, key):
    prompt, last = script_prompt(niche, topic, n, voice), ""
    key = key.strip().strip('"').strip("'")
    for m in GEMINI_MODELS:
        r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent",
                          headers={"x-goog-api-key": key}, timeout=60,
                          json={"contents": [{"parts": [{"text": prompt}]}],
                                "generationConfig": {"responseMimeType": "application/json"}})
        if r.status_code == 200:
            text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(re.search(r"\{.*\}", text, re.S).group(0))
        last = f"{m}: {r.status_code} {r.text[:150]}"
    raise RuntimeError("Gemini failed - " + last)


def get_script(niche, topic, n, voice, key):
    import anthropic
    msg = anthropic.Anthropic(api_key=key).messages.create(
        model=MODEL, max_tokens=2000, messages=[{"role": "user", "content": script_prompt(niche, topic, n, voice)}])
    return json.loads(re.search(r"\{.*\}", msg.content[0].text, re.S).group(0))


def parse_manual(text):
    scenes = []
    for line in text.strip().splitlines():
        if line.strip():
            narr, _, kw = line.partition("|")
            scenes.append({"narration": narr.strip(), "keywords": kw.strip() or "nature landscape"})
    return {"title": "", "description": "", "scenes": scenes}


PRESETS = {"Normal": (0, 0, 0), "Deep and slow": (-15, -25, 0), "Fast and energetic": (25, 10, 10),
           "Calm": (-10, -5, -10), "Excited": (15, 15, 10), "Storyteller": (-5, -10, 0)}  # speed %, pitch Hz, volume %


def tts(text, voice, path, rate=0, pitch=0, vol=0):
    try:
        asyncio.run(edge_tts.Communicate(text, voice, rate=f"{rate:+d}%", pitch=f"{pitch:+d}Hz",
                                         volume=f"{vol:+d}%").save(path))
        if os.path.getsize(path) < 1000:
            raise RuntimeError("empty audio")
    except Exception:
        gTTS(text, lang=voice[:2]).save(path)


def fetch_clip(query, key, folder, used, errs):
    try:
        key = key.strip().strip('"').strip("'")
        resp = requests.get("https://api.pexels.com/videos/search", headers={"Authorization": key},
                            params={"query": query, "orientation": "portrait", "per_page": 10}, timeout=20)
        if resp.status_code != 200:
            errs.append(f"Pexels replied {resp.status_code}: {resp.text[:120]}")
            return None
        for v in resp.json().get("videos", []):
            if v["id"] in used:
                continue
            files = sorted([f for f in v["video_files"] if f.get("file_type") == "video/mp4"],
                           key=lambda f: f.get("height", 0))
            good = [f for f in files if f.get("height", 0) >= 1280] or files[-1:]
            if good:
                used.add(v["id"])
                path = os.path.join(folder, f"{v['id']}.mp4")
                with open(path, "wb") as fh:
                    fh.write(requests.get(good[0]["link"], timeout=90).content)
                return path
        errs.append(f"No Pexels results for '{query}'")
    except Exception as e:
        errs.append(f"Pexels request failed: {e}")
    return None


def fetch_pixabay(query, key, folder, used, errs, vtype="all"):
    try:
        key = key.strip().strip('"').strip("'")
        resp = requests.get("https://pixabay.com/api/videos/", timeout=20,
                            params={"key": key, "q": query[:90], "per_page": 15, "safesearch": "true",
                                    "video_type": vtype})
        if resp.status_code != 200:
            errs.append(f"Pixabay replied {resp.status_code}: {resp.text[:120]}")
            return None
        for h in resp.json().get("hits", []):
            if ("px", h["id"]) in used:
                continue
            vids = h.get("videos", {})
            f = next((vids[k] for k in ("large", "medium", "small") if vids.get(k, {}).get("url")), None)
            if f:
                used.add(("px", h["id"]))
                path = os.path.join(folder, f"px{h['id']}.mp4")
                with open(path, "wb") as fh:
                    fh.write(requests.get(f["url"], timeout=90).content)
                return path
        errs.append(f"No Pixabay results for '{query}'")
    except Exception as e:
        errs.append(f"Pixabay request failed: {e}")
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


def caption(text, dur, W=W, H=H):
    k = W / 1080
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d, font = ImageDraw.Draw(img), load_font(max(18, int(58 * k)), text)
    lines, line = [], ""
    for w in text.split():
        t = (line + " " + w).strip()
        if d.textlength(t, font=font) > W - int(160 * k):
            lines.append(line)
            line = w
        else:
            line = t
    lines.append(line)
    step = int(80 * k)
    y = int(H * 0.73) - len(lines) * step // 2
    for l in lines:
        d.text(((W - d.textlength(l, font=font)) / 2, y), l, font=font, fill="white",
               stroke_width=max(2, int(5 * k)), stroke_fill="black")
        y += step
    return ImageClip(np.array(img)).set_duration(dur)


def build(scenes, pexels_key, voice, captions, out, log, errs, mod=(0, 0, 0), size=(1080, 1920), vtype="all"):
    W, H = size
    tmp, clips, used, missed = tempfile.mkdtemp(), [], set(), 0
    if vtype == "animation" and not secret("PIXABAY_API_KEY"):
        errs.append("Animation style needs PIXABAY_API_KEY in Secrets")
    if not pexels_key and not secret("PIXABAY_API_KEY"):
        errs.append("No PEXELS_API_KEY or PIXABAY_API_KEY found in Secrets")
    for i, s in enumerate(scenes):
        log(f"Scene {i + 1}/{len(scenes)}: voice and footage...")
        mp3 = os.path.join(tmp, f"v{i}.mp3")
        tts(s["narration"], voice, mp3, *mod)
        audio = AudioFileClip(mp3)
        dur = audio.duration + 0.3
        q = s["keywords"] if s["keywords"].isascii() else "nature landscape"
        path = fetch_clip(q, pexels_key, tmp, used, errs) if pexels_key and vtype != "animation" else None
        if not path and secret("PIXABAY_API_KEY"):
            path = fetch_pixabay(q, secret("PIXABAY_API_KEY"), tmp, used, errs, vtype)
        if path:
            v = VideoFileClip(path).without_audio()
            v = v.fx(vfx.loop, duration=dur) if v.duration < dur else v.subclip(0, dur)
            v = v.resize(max(W / v.w, H / v.h))
            v = v.crop(x_center=v.w / 2, y_center=v.h / 2, width=W, height=H)
        else:
            missed += 1
            v = ColorClip((W, H), color=(20, 20, 40)).set_duration(dur)
        layers = [v] + ([caption(s["narration"], dur, W, H)] if captions else [])
        clips.append(CompositeVideoClip(layers, size=(W, H)).set_audio(audio))
    log("Rendering final video (1 to 3 minutes)...")
    concatenate_videoclips(clips, method="compose").write_videofile(
        out, fps=24, codec="libx264", audio_codec="aac", preset="ultrafast", bitrate=BITRATE[W], threads=2, logger=None)
    return missed


st.set_page_config(page_title="AutoShort AI", page_icon="🎬", layout="centered")
st.title("🎬 AutoShort AI")
claude_key, pexels_key = secret("ANTHROPIC_API_KEY"), secret("PEXELS_API_KEY")
gemini_key = secret("GEMINI_API_KEY")
if not pexels_key and not secret("PIXABAY_API_KEY"):
    st.warning("No video key found (PIXABAY_API_KEY or PEXELS_API_KEY), so videos will have a plain dark background. Add one in app Settings > Secrets.")

niche = st.selectbox("Niche", NICHES)
topic = st.text_input("Topic (optional)", placeholder="Leave empty and AI picks one")
lang_pick = st.radio("Voice language", ["Tamil", "English", "Hindi"], horizontal=True)
code = {"Tamil": "ta", "English": "en", "Hindi": "hi"}[lang_pick]
choices = {k: v for k, v in load_voices().items() if v.startswith(code)} or \
          {k: v for k, v in VOICES.items() if v.startswith(code)}
voice_name = st.selectbox(f"Voice ({len(choices)} available)", list(choices))


def apply_preset():
    r, p, v = PRESETS[st.session_state["preset"]]
    st.session_state.update(rate=r, pitch=p, vol=v)


with st.expander("🎚 Voice modulation"):
    st.selectbox("Style preset", list(PRESETS), key="preset", on_change=apply_preset)
    rate = st.slider("Speed (%)", -50, 100, 0, key="rate")
    pitch = st.slider("Pitch (Hz)", -50, 50, 0, key="pitch")
    vol = st.slider("Volume (%)", -50, 50, 0, key="vol")
mod = (rate, pitch, vol)
if st.button("▶ Preview voice"):
    sample = {"ta": "வணக்கம்! இது என் குரலின் மாதிரி.", "en": "Hello! This is a sample of my voice.",
              "hi": "नमस्ते! यह मेरी आवाज़ का नमूना है।"}[code]
    p = os.path.join(tempfile.mkdtemp(), "preview.mp3")
    tts(sample, choices[voice_name], p, *mod)
    st.audio(p)
c1, c2 = st.columns(2)
n = c1.slider("Scenes", 3, 8, 5)
caps = c2.checkbox("Captions", True)
quality = st.selectbox("Video quality (MP4)", list(QUALITY), index=1)
VTYPE = {"Real footage": "film", "Animation / cartoon": "animation", "Any": "all"}
style = st.selectbox("Video style", list(VTYPE))
manual = st.text_area("Or paste your own script", placeholder="One scene per line: narration | keywords")

if st.button("Generate video", type="primary", use_container_width=True):
    try:
        with st.status("Working...", expanded=True) as status:
            if manual.strip():
                data = parse_manual(manual)
            elif gemini_key:
                status.write("Writing script with Gemini...")
                data = get_script_gemini(niche, topic, n, choices[voice_name], gemini_key)
            elif claude_key:
                status.write("Writing script with Claude...")
                data = get_script(niche, topic, n, choices[voice_name], claude_key)
            else:
                st.error("No GEMINI_API_KEY found in Secrets. Add it, or paste your own script.")
                st.stop()
            out = os.path.join(tempfile.mkdtemp(), "short.mp4")
            errs = []
            missed = build(data["scenes"], pexels_key, choices[voice_name], caps, out, status.write, errs, mod, QUALITY[quality], VTYPE[style])
            if not missed:
                errs = []
            status.update(label="Done", state="complete")
        st.session_state.update(video=open(out, "rb").read(), data=data, missed=missed, errs=errs)
    except Exception as e:
        st.error("Something failed. Send this error to Claude:")
        st.exception(e)

if "video" in st.session_state:
    d = st.session_state["data"]
    if st.session_state.get("missed"):
        st.info(f"{st.session_state['missed']} scene(s) had no stock clip and use a plain background.")
        for e in sorted(set(st.session_state.get("errs", []))):
            st.error(e)
    st.video(st.session_state["video"])
    st.download_button("Download MP4", st.session_state["video"], "short.mp4", "video/mp4", use_container_width=True)
    if d.get("title"):
        st.subheader("Title")
        st.code(d["title"], language=None)
        st.subheader("Description")
        st.code(d["description"], language=None)
