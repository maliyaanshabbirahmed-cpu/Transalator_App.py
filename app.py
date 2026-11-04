import io
import os
import html
import tempfile

import streamlit as st
from faster_whisper import WhisperModel
from gtts import gTTS
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import torch

st.set_page_config(
    page_title="Voice Translator",
    page_icon="🎙️",
    layout="centered",
)

LANGUAGES = {
    "English": {"m2m": "en", "whisper": "en", "tts": "en"},
    "Urdu": {"m2m": "ur", "whisper": "ur", "tts": "ur"},
    "Hindi": {"m2m": "hi", "whisper": "hi", "tts": "hi"},
    "Arabic": {"m2m": "ar", "whisper": "ar", "tts": "ar"},
    "Punjabi": {"m2m": "pa", "whisper": "pa", "tts": "pa"},
}

TRANSLATION_MODEL = "facebook/m2m100_418M"
ASR_MODEL = os.getenv("WHISPER_MODEL", "base")


@st.cache_resource(show_spinner="Loading speech recognition model...")
def load_asr():
    return WhisperModel(
        ASR_MODEL,
        device="cpu",
        compute_type="int8",
    )


@st.cache_resource(show_spinner="Loading translation model...")
def load_translation():
    tokenizer = M2M100Tokenizer.from_pretrained(TRANSLATION_MODEL)
    model = M2M100ForConditionalGeneration.from_pretrained(TRANSLATION_MODEL)
    model.eval()
    return tokenizer, model


def transcribe_audio(audio_bytes: bytes, language_code: str) -> str:
    model = load_asr()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_file:
        audio_file.write(audio_bytes)
        audio_path = audio_file.name

    try:
        segments, _ = model.transcribe(
            audio_path,
            language=language_code,
            beam_size=5,
            vad_filter=True,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return text
    finally:
        try:
            os.remove(audio_path)
        except OSError:
            pass


def translate_text(text: str, source_code: str, target_code: str) -> str:
    tokenizer, model = load_translation()
    tokenizer.src_lang = source_code

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.get_lang_id(target_code),
            max_new_tokens=256,
            num_beams=4,
        )

    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()


def text_to_speech(text: str, language_code: str) -> bytes:
    audio_buffer = io.BytesIO()
    speech = gTTS(text=text, lang=language_code, slow=False)
    speech.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    return audio_buffer.read()


st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 2.35rem;
        font-weight: 800;
        letter-spacing: -1px;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        text-align: center;
        color: #8b929e;
        margin-bottom: 1.8rem;
    }
    div[data-testid="stAudioInput"] {
        border: 1px solid rgba(128,128,128,.35);
        border-radius: 14px;
        padding: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">🎙️ Voice Translator</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Speak naturally. Get your translation spoken back.</div>',
    unsafe_allow_html=True,
)

left, right = st.columns(2)
with left:
    source_language = st.selectbox("Speak in", list(LANGUAGES.keys()), index=0)
with right:
    target_options = [name for name in LANGUAGES if name != source_language]
    target_language = st.selectbox("Translate to", target_options, index=0)

st.caption("Select your languages, record a short message, and press Translate.")

recording = st.audio_input("Record your voice", sample_rate=16000)

if recording is not None:
    st.audio(recording, format="audio/wav")

translate_clicked = st.button(
    "Translate and speak",
    type="primary",
    use_container_width=True,
    disabled=recording is None,
)

if translate_clicked and recording is not None:
    source_code = LANGUAGES[source_language]["m2m"]
    target_code = LANGUAGES[target_language]["m2m"]

    try:
        with st.spinner("Recognizing your speech..."):
            recognized_text = transcribe_audio(
                recording.getvalue(),
                LANGUAGES[source_language]["whisper"],
            )

        if not recognized_text:
            st.error("No speech was detected. Please record again, speaking clearly.")
            st.stop()

        st.subheader("What you said")
        st.write(recognized_text)

        with st.spinner("Translating..."):
            translated_text = translate_text(
                recognized_text,
                source_code,
                target_code,
            )

        if not translated_text:
            st.error("The translation model returned an empty result. Please try again.")
            st.stop()

        st.subheader(f"Translation — {target_language}")
        st.markdown(
            f'<div style="padding:16px;border:1px solid rgba(128,128,128,.35);'
            f'border-radius:12px;font-size:1.15rem;direction:'
            f'{"rtl" if target_language in ["Urdu", "Arabic"] else "ltr"}">'
            f'{html.escape(translated_text)}</div>',
            unsafe_allow_html=True,
        )

        with st.spinner("Generating translated voice..."):
            spoken_audio = text_to_speech(
                translated_text,
                LANGUAGES[target_language]["tts"],
            )

        st.subheader("Listen to the translation")
        st.audio(spoken_audio, format="audio/mp3")

        st.download_button(
            "Download translated audio",
            data=spoken_audio,
            file_name="translated_speech.mp3",
            mime="audio/mpeg",
            use_container_width=True,
        )

    except Exception as error:
        st.error(
            "The translation could not be completed. Check your internet connection "
            "and try a shorter recording. If the problem continues, see the troubleshooting "
            "section in README.md."
        )
        with st.expander("Technical details"):
            st.code(str(error))

st.divider()
st.caption(
    "Prototype notice: machine translation and speech recognition can make mistakes, "
    "especially with regional accents, noisy recordings, names, and mixed-language speech. "
    "Do not rely on this app for medical, legal, emergency, or other critical communication."
)
