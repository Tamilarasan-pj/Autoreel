from urllib.parse import quote
import os, json, re, time, asyncio, tempfile, requests, threading, subprocess, shutil
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import streamlit as st
import edge_tts
from gtts import gTTS
from PIL import Image, ImageDraw, ImageFont

# ── constants ────────────────────────────────────────────────────────────────
W, H       = 1080, 1920
QUALITY    = {"480p (fast)": (480, 854), "720p (HD)": (720, 1280), "1080p (Full HD)": (1080, 1920)}
BITRATE    = {480: "2000k", 720: "4500k", 1080: "8000k"}
VOICES_FB  = {"English (Aria, female)": "en-US-AriaNeural",
              "English (Guy, male)":    "en-US-GuyNeural",
              "Tamil (Pallavi, female)":"ta-IN-PallaviNeural",
              "Tamil (Valluvar, male)": "ta-IN-ValluvarNeural",
              "Hindi (Swara, female)":  "hi-IN-SwaraNeural"}
NICHES     = ["Amazing facts","Money tips","History mysteries","Motivation","Scary stories","AI & tech news"]
PRESETS    = {"Normal":(0,0,0),"Deep & slow":(-15,-25,0),"Fast & energetic":(25,10,10),
              "Calm":(-10,-5,-10),"Excited":(15,15,10),"Storyteller":(-5,-10,0)}
GEMINI_M   = ["gemini-2.5-flash","gemini-2.5-flash-lite","gemini-2.0-flash",
              "gemini-flash-latest","gemini-flash-lite-latest"]
IMG_STYLES = {
    "Cinematic":    "cinematic, photorealistic, dramatic lighting, 4k",
    "Cartoon":      "colorful cartoon animation, bold outlines, vibrant",
    "Anime":        "anime style, detailed, vibrant colors",
    "3D Render":    "3D render, pixar style, soft lighting",
    "Dark Mystery": "dark moody atmosphere, mysterious, fog",
    "Watercolor":   "watercolor painting, soft pastel colors"
}
FONTS       = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "DejaVuSans-Bold.ttf"]
TAMIL_FONTS = ["/usr/share/fonts/truetype/noto/NotoSansTamil-Bold.ttf",
               "/usr/share/fonts/truetype/noto/NotoSansTamil-Regular.ttf"]
IMG_LOCK, IMG_LAST = threading.Lock(), [0.0]


# ── helpers ───────────────────────────────────────────────────────────────────
def secret(k):
    try:    return st.secrets[k]
    except: return os.getenv(k, "")


def get_ffmpeg():
    e = shutil.which("ffmpeg")
    if not e:
        import imageio_ffmpeg
        e = imageio_ffmpeg.get_ffmpeg_exe()
    return e


def get_ffprobe():
    return shutil.which("ffprobe") or get_ffmpeg().replace("ffmpeg", "ffprobe")


def run_ff(args, timeout=300):
    p = subprocess.run([get_ffmpeg(), "-y"] + args,
                       capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError("ffmpeg: " + p.stderr[-300:])


def get_dur(path):
    r = subprocess.run([get_ffprobe(), "-v", "error", "-show_entries",
                        "format=duration", "-of", "csv=p=0", path],
                       capture_output=True, text=True, timeout=30)
    return float(r.stdout.strip())


# ── voices ────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=86400)
def load_voices():
    try:
        out = {}
        for v in sorted(asyncio.run(edge_tts.list_voices()),
                        key=lambda v: (v["Locale"], v["ShortName"])):
            if v["Locale"][:2] in ("ta", "en", "hi"):
                name = v["ShortName"].split("-", 2)[2].replace("Neural", "")
                out[f'{name} | {v["Gender"]} | {v["Locale"]}'] = v["ShortName"]
        return out or VOICES_FB
    except Exception:
        return VOICES_FB


# ── TTS ───────────────────────────────────────────────────────────────────────
def tts(text, voice, path, rate=0, pitch=0, vol=0):
    try:
        asyncio.run(edge_tts.Communicate(
            text, voice,
            rate=f"{rate:+d}%", pitch=f"{pitch:+d}Hz", volume=f"{vol:+d}%"
        ).save(path))
        if os.path.getsize(path) < 1000:
            raise RuntimeError("empty audio")
    except Exception:
        gTTS(text, lang=voice[:2]).save(path)


# ── script ────────────────────────────────────────────────────────────────────
def script_prompt(niche, topic, n, voice):
    lang = "Tamil" if voice.startswith("ta") else "Hindi" if voice.startswith("hi") else "English"
    return (
        f"Create a YouTube Short script. Niche: '{niche}'. "
        f"Topic: {topic or 'pick a fresh, surprising topic'}. "
        f"Language: {lang}. Exactly {n} scenes. "
        "Scene 1 must be a strong hook that makes viewers stop scrolling. "
        "Each scene: punchy 1-2 sentence narration with natural commas and pauses, "
        "2-3 English stock-footage keywords (visual things like 'ocean waves'), "
        "and an image_prompt: vivid English scene description, no text in image. "
        'Return ONLY valid JSON: {"title":"...","description":"... #hashtags",'
        '"scenes":[{"narration":"...","keywords":"...","image_prompt":"..."}]}'
    )


def get_script_gemini(niche, topic, n, voice, key):
    prompt, errors = script_prompt(niche, topic, n, voice), []
    key = key.strip().strip('"').strip("'")
    for attempt in range(3):
        busy = False
        for m in GEMINI_M:
            try:
                r = requests.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent",
                    headers={"x-goog-api-key": key}, timeout=60,
                    json={"contents": [{"parts": [{"text": prompt}]}],
                          "generationConfig": {"responseMimeType": "application/json"}})
            except Exception as e:
                errors.append(f"{m}: {e}"); busy = True; continue
            if r.status_code == 200:
                try:
                    text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
                    return json.loads(re.search(r"\{.*\}", text, re.S).group(0))
                except Exception as e:
                    errors.append(f"{m}: bad reply ({e})"); continue
            errors.append(f"{m}: {r.status_code}")
            if r.status_code in (429, 500, 503):
                busy = True
        if not busy:
            break
        time.sleep(3 * (attempt + 1))
    raise RuntimeError("Gemini busy. Try again or paste your own script. " +
                       ", ".join(errors[-5:]))


def parse_manual(text):
    scenes = []
    for line in text.strip().splitlines():
        if line.strip():
            narr, _, kw = line.partition("|")
            scenes.append({
                "narration": narr.strip(),
                "keywords":  kw.strip() or "nature landscape",
                "image_prompt": narr.strip()
            })
    return {"title": "", "description": "", "scenes": scenes}


# ── stock footage ─────────────────────────────────────────────────────────────
USED_LOCK = threading.Lock()


def claim(used, ident):
    with USED_LOCK:
        if ident in used:
            return False
        used.add(ident)
        return True


def fetch_pixabay(query, key, folder, used, errs, vtype="all", target_w=720):
    prefs = ("large", "medium", "small") if target_w >= 1080 else ("medium", "small", "large")
    try:
        key = key.strip().strip('"').strip("'")
        r = requests.get("https://pixabay.com/api/videos/", timeout=20,
                         params={"key": key, "q": query[:90], "per_page": 15,
                                 "safesearch": "true", "video_type": vtype})
        if r.status_code != 200:
            errs.append(f"Pixabay {r.status_code}: {r.text[:100]}")
            return None
        for h in r.json().get("hits", []):
            vids = h.get("videos", {})
            f = next((vids[k] for k in prefs if vids.get(k, {}).get("url")), None)
            if f and claim(used, ("px", h["id"])):
                path = os.path.join(folder, f"px{h['id']}.mp4")
                with open(path, "wb") as fh:
                    fh.write(requests.get(f["url"], timeout=90).content)
                return path
        errs.append(f"No Pixabay results for '{query}'")
    except Exception as e:
        errs.append(f"Pixabay error: {e}")
    return None


# ── AI images ─────────────────────────────────────────────────────────────────
def gen_image(prompt, W, H, seed, folder, errs):
    rw, rh = (720, 1280) if W >= 720 else (480, 854)
    url = "https://image.pollinations.ai/prompt/" + quote(prompt[:600])
    for attempt in range(4):
        with IMG_LOCK:
            wait = 12 - (time.time() - IMG_LAST[0])
            if wait > 0:
                time.sleep(wait)
            try:
                r = requests.get(url, timeout=120,
                                 params={"width": rw, "height": rh,
                                         "seed": seed + attempt,
                                         "nologo": "true", "model": "flux"})
            except Exception as e:
                IMG_LAST[0] = time.time()
                errs.append(f"AI image failed: {e}")
                continue
            IMG_LAST[0] = time.time()
        if (r.status_code == 200 and
                r.headers.get("content-type", "").startswith("image") and
                len(r.content) > 5000):
            path = os.path.join(folder, f"ai{seed}.jpg")
            with open(path, "wb") as fh:
                fh.write(r.content)
            return path
        errs.append(f"AI image service {r.status_code}")
    return None


# ── ffmpeg clip utils ─────────────────────────────────────────────────────────
def fit_clip(src, dst, dur, W, H):
    run_ff(["-stream_loop", "-1", "-i", src, "-t", f"{dur:.3f}",
            "-vf", f"scale={W}:{H}:force_original_aspect_ratio=increase,"
                   f"crop={W}:{H},fps=24",
            "-an", "-c:v", "libx264", "-preset", "ultrafast",
            "-crf", "26", "-pix_fmt", "yuv420p", dst])


def ken_burns(img, dst, dur, W, H, idx=0):
    frames = max(2, round(dur * 24))
    inc    = 0.25 / frames
    bw = int(W * 1.5) // 2 * 2
    bh = int(H * 1.5) // 2 * 2
    z = (f"min(zoom+{inc:.6f},1.25)" if idx % 2 == 0
         else f"if(lte(zoom,1.0),1.25,max(1.0,zoom-{inc:.6f}))")
    run_ff(["-i", img, "-vf",
            f"scale={bw}:{bh}:force_original_aspect_ratio=increase,"
            f"crop={bw}:{bh},"
            f"zoompan=z='{z}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
            f"d={frames}:s={W}x{H}:fps=24,format=yuv420p",
            "-frames:v", str(frames), "-an",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "26", dst])


def make_blank(dst, dur, W, H):
    run_ff(["-f", "lavfi", "-i", f"color=c=0x141428:size={W}x{H}:rate=24",
            "-t", f"{dur:.3f}", "-an", "-c:v", "libx264",
            "-preset", "ultrafast", "-crf", "26", "-pix_fmt", "yuv420p", dst])


# ── captions via ffmpeg drawtext ──────────────────────────────────────────────
def load_font_path(is_tamil=False):
    for f in (TAMIL_FONTS if is_tamil else []) + FONTS:
        if os.path.exists(f):
            return f
    return ""


def caption_filter(text, W, H, is_tamil=False):
    safe = text.replace("'", "\u2019").replace(":", r"\:").replace(",", "\\,")
    fp   = load_font_path(is_tamil)
    size = max(18, int(58 * (W / 1080)))
    y    = int(H * 0.73)
    font_arg = f":fontfile='{fp}'" if fp else ""
    bw = max(2, int(5 * (W / 1080)))
    return (f"drawtext=text='{safe}'{font_arg}:fontsize={size}"
            f":fontcolor=white:bordercolor=black:borderw={bw}"
            f":x=(w-text_w)/2:y={y}:line_spacing=8")


# ── background music ──────────────────────────────────────────────────────────
def make_bgm(dst, dur, vol_db=-18):
    sr, freq, amp = 44100, 220, 0.12
    n  = int(sr * max(dur + 1, 1))
    t  = np.linspace(0, n / sr, n, False)
    sig = (amp * (np.sin(2 * np.pi * freq * t) +
                  0.4 * np.sin(2 * np.pi * freq * 1.5 * t) +
                  0.2 * np.sin(2 * np.pi * freq * 2.0 * t))).astype(np.float32)
    fade = int(sr * 0.5)
    sig[:fade]  *= np.linspace(0, 1, fade)
    sig[-fade:] *= np.linspace(1, 0, fade)
    raw = dst + ".raw"
    with open(raw, "wb") as f:
        f.write((sig * 32767).astype(np.int16).tobytes())
    run_ff(["-f", "s16le", "-ar", str(sr), "-ac", "1", "-i", raw,
            "-af", f"volume={vol_db}dB", "-t", f"{dur:.3f}", dst])
    os.remove(raw)


# ── core build ────────────────────────────────────────────────────────────────
def build(scenes, pix_key, voice, do_caps, out, log, errs,
          mod=(0, 0, 0), size=(720, 1280), vtype="all",
          visuals="stock", img_style="", do_music=True):
    W, H = size
    tmp, used = tempfile.mkdtemp(), set()
    is_tamil  = voice.startswith("ta")

    def prep(i, s):
        mp3 = os.path.join(tmp, f"v{i}.mp3")
        tts(s["narration"], voice, mp3, *mod)
        dur = get_dur(mp3) + 0.3
        fitted = None

        # AI images
        if visuals == "ai":
            prompt = (f'{s.get("image_prompt") or s["keywords"]}, '
                      f'{img_style}, vertical composition, no text, no watermark')
            img = gen_image(prompt, W, H, int(time.time()) % 100000 + i * 7, tmp, errs)
            if img:
                try:
                    fitted = os.path.join(tmp, f"f{i}.mp4")
                    ken_burns(img, fitted, dur, W, H, i)
                except Exception as e:
                    errs.append(f"Ken Burns failed: {e}")
                    fitted = None

        # Stock video
        if not fitted:
            q = s["keywords"] if s["keywords"].isascii() else "nature landscape"
            path = fetch_pixabay(q, pix_key, tmp, used, errs, vtype, W) if pix_key else None
            if path:
                try:
                    fitted = os.path.join(tmp, f"f{i}.mp4")
                    fit_clip(path, fitted, dur, W, H)
                except Exception as e:
                    errs.append(f"Clip resize: {e}")
                    fitted = None

        # Fallback blank
        if not fitted:
            fitted = os.path.join(tmp, f"f{i}.mp4")
            make_blank(fitted, dur, W, H)

        return mp3, dur, fitted

    log(f"⚡ Preparing {len(scenes)} scenes in parallel…")
    with ThreadPoolExecutor(max_workers=min(4, len(scenes))) as ex:
        results = list(ex.map(lambda a: prep(*a), enumerate(scenes)))

    scene_files = []
    for i, (sc, (mp3, dur, fitted)) in enumerate(zip(scenes, results)):
        log(f"🎬 Compositing scene {i + 1}/{len(scenes)}…")
        scene_out = os.path.join(tmp, f"scene{i}.mp4")

        if do_caps:
            cap_out = os.path.join(tmp, f"cap{i}.mp4")
            cf = caption_filter(sc["narration"], W, H, is_tamil)
            run_ff(["-i", fitted, "-vf", cf, "-an",
                    "-c:v", "libx264", "-preset", "ultrafast",
                    "-crf", "26", "-pix_fmt", "yuv420p", cap_out])
            vid_src = cap_out
        else:
            vid_src = fitted

        run_ff(["-i", vid_src, "-i", mp3,
                "-c:v", "copy", "-c:a", "aac", "-shortest", scene_out])
        scene_files.append(scene_out)

    log("🔗 Joining scenes…")
    joined = os.path.join(tmp, "joined.mp4")
    lst    = os.path.join(tmp, "list.txt")
    with open(lst, "w") as f:
        for sf in scene_files:
            f.write(f"file '{sf}'\n")
    run_ff(["-f", "concat", "-safe", "0", "-i", lst,
            "-c:v", "libx264", "-c:a", "aac",
            "-preset", "ultrafast", "-b:v", BITRATE[W],
            "-movflags", "+faststart", joined])

    if do_music:
        log("🎵 Mixing background music…")
        total = sum(d for _, d, _ in results)
        bgm   = os.path.join(tmp, "bgm.wav")
        try:
            make_bgm(bgm, total)
            run_ff(["-i", joined, "-i", bgm,
                    "-filter_complex",
                    "[0:a]volume=1.0[va];[1:a]volume=0.15[vb];[va][vb]amix=inputs=2:duration=first[ao]",
                    "-map", "0:v", "-map", "[ao]",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
                    "-movflags", "+faststart", out])
        except Exception as e:
            errs.append(f"Music mix failed (video still saved): {e}")
            shutil.copy(joined, out)
    else:
        shutil.copy(joined, out)

    missed = sum(1 for _, _, f in results if "blank" in os.path.basename(f))
    return missed


# ── UI ─────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="AutoShort AI", page_icon="🎬", layout="centered")
st.markdown("""
<style>
[data-testid="stAppViewContainer"]{background:#0d0d0d}
.main .block-container{padding-top:1.5rem}
h1{color:#ff4444!important;font-size:2rem!important}
.stButton>button{background:#ff4444!important;color:white!important;
  border:none!important;border-radius:8px!important;font-weight:bold!important}
.stButton>button:hover{background:#cc0000!important}
.stTabs [data-baseweb="tab"]{font-size:1rem}
</style>""", unsafe_allow_html=True)

st.title("🎬 AutoShort AI")
st.caption("Create faceless YouTube Shorts in minutes — free, fully automated")

gemini_key = secret("GEMINI_API_KEY")
pix_key    = secret("PIXABAY_API_KEY")

if not pix_key:
    st.warning("⚠️ Add PIXABAY_API_KEY in Secrets for stock footage. Videos will use plain backgrounds otherwise.")

tab_create, tab_tips = st.tabs(["✨ Create", "💡 Tips & Settings"])

# ── TIPS TAB ──────────────────────────────────────────────────────────────────
with tab_tips:
    st.subheader("Required Secrets")
    st.info("Manage app → Settings → Secrets")
    st.code('GEMINI_API_KEY  = "your-key-here"\nPIXABAY_API_KEY = "your-key-here"', language="toml")
    st.subheader("Speed tips")
    st.markdown("""
- Use **480p** for tests, **720p** for real uploads
- Keep **3–4 scenes** for fastest render
- Shorter narration = faster voice = shorter clips
- Turn off **Background music** if the render is slow
""")
    st.subheader("Better voice")
    st.markdown("""
- Use commas and full stops — edge-tts follows punctuation
- Exclamation marks add energy
- Short sentences sound more natural than long ones
""")
    st.subheader("Better visuals")
    st.markdown("""
- Write keywords in English even for Tamil videos
- Specific keywords give better clips: "underwater coral reef" beats "ocean"
- AI images look more consistent than mixed stock clips
""")

# ── CREATE TAB ────────────────────────────────────────────────────────────────
with tab_create:
    # Niche + Topic
    c1, c2 = st.columns(2)
    with c1:
        niche = st.selectbox("Niche", NICHES + ["Other (type my own)"])
        if niche.startswith("Other"):
            niche = st.text_input("Your niche", placeholder="e.g. Tamil history") or "Amazing facts"
    with c2:
        topic = st.text_input("Topic (optional)", placeholder="Leave empty — AI picks one")

    # Voice
    lang_pick = st.radio("Voice language", ["Tamil", "English", "Hindi"], horizontal=True)
    code      = {"Tamil": "ta", "English": "en", "Hindi": "hi"}[lang_pick]
    choices   = ({k: v for k, v in load_voices().items() if v.startswith(code)}
                 or {k: v for k, v in VOICES_FB.items() if v.startswith(code)})
    voice_name = st.selectbox(f"Voice ({len(choices)} available)", list(choices))

    with st.expander("🎚 Voice modulation"):
        def apply_preset():
            r, p, v = PRESETS[st.session_state["preset"]]
            st.session_state.update(rate=r, pitch=p, vol=v)
        st.selectbox("Preset", list(PRESETS), key="preset", on_change=apply_preset)
        rate  = st.slider("Speed (%)",  -50, 100, 0, key="rate")
        pitch = st.slider("Pitch (Hz)", -50,  50, 0, key="pitch")
        vol   = st.slider("Volume (%)", -50,  50, 0, key="vol")
    mod = (rate, pitch, vol)

    if st.button("▶ Preview voice"):
        sample = {"ta": "வணக்கம்! இது என் குரலின் மாதிரி.",
                  "en": "Hello! This is a sample of my voice.",
                  "hi": "नमस्ते! यह मेरी आवाज़ का नमूना है।"}[code]
        pv = os.path.join(tempfile.mkdtemp(), "preview.mp3")
        tts(sample, choices[voice_name], pv, *mod)
        st.audio(pv)

    # Video settings
    ca, cb, cc = st.columns(3)
    n       = ca.slider("Scenes", 3, 8, 5)
    quality = cb.selectbox("Quality", list(QUALITY), index=1)
    caps    = cc.checkbox("Captions", True)
    music   = st.checkbox("🎵 Background music", True)

    # Visuals
    visuals = st.radio("Visuals", ["Stock video", "AI images (slower, free)"], horizontal=True)
    if visuals.startswith("AI"):
        img_style = st.selectbox("Image style", list(IMG_STYLES))
        vtype     = "all"
    else:
        img_style = None
        clip_type = st.selectbox("Clip type", ["Any", "Real footage", "Animation"])
        vtype     = {"Any": "all", "Real footage": "film", "Animation": "animation"}[clip_type]

    # Manual script
    manual = st.text_area("Or paste your own script",
                          placeholder="One line per scene: narration text | english keywords")

    st.divider()

    if st.button("🎬 Generate video", type="primary", use_container_width=True):
        if not gemini_key and not manual.strip():
            st.error("Add GEMINI_API_KEY in Secrets, or paste your own script below.")
        else:
            try:
                with st.status("Working…", expanded=True) as status:
                    if manual.strip():
                        data = parse_manual(manual)
                    else:
                        status.write("✍️ Writing script with Gemini…")
                        data = get_script_gemini(niche, topic, n,
                                                 choices[voice_name], gemini_key)
                    out  = os.path.join(tempfile.mkdtemp(), "short.mp4")
                    errs = []
                    build(
                        data["scenes"], pix_key, choices[voice_name],
                        caps, out, status.write, errs, mod,
                        QUALITY[quality], vtype,
                        "ai" if visuals.startswith("AI") else "stock",
                        IMG_STYLES.get(img_style, ""), music
                    )
                    status.update(label="✅ Done!", state="complete")
                st.session_state.update(
                    video=open(out, "rb").read(), data=data, errs=errs)
            except Exception as e:
                st.error("Something went wrong — send this error to Claude:")
                st.exception(e)

# ── result ────────────────────────────────────────────────────────────────────
if "video" in st.session_state:
    d = st.session_state["data"]
    for e in sorted(set(st.session_state.get("errs", []))):
        st.error(e)
    st.video(st.session_state["video"])
    st.download_button("⬇️ Download MP4",
                       st.session_state["video"],
                       file_name="short.mp4",
                       mime="video/mp4",
                       use_container_width=True)
    if d.get("title"):
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**📋 YouTube Title**")
            st.code(d["title"], language=None)
        with col2:
            st.markdown("**📋 YouTube Description**")
            st.code(d["description"], language=None)
        st.caption("Copy the title and description above when uploading to YouTube.")
