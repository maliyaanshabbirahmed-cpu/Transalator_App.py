# Voice Translator

A simple Streamlit voice translator. The user selects a source and target language, records speech using the microphone, and receives translated text plus spoken audio.

## Supported languages

- English
- Urdu
- Hindi
- Arabic
- Punjabi

The translation model is Meta's `facebook/m2m100_418M`, which supports direct many-to-many translation across 100 languages and is published under the MIT license.

Speech recognition uses Faster-Whisper (`base` by default). Text-to-speech uses gTTS.

## Project files

```text
voice_translator_app/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── .streamlit/
    └── config.toml
```

## Run locally

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
```

Activate the environment:

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install dependencies and start the app:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The first run downloads the speech and translation models. This requires an internet connection and can take several minutes. The translation model is relatively large and runs on CPU.

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload all files and folders from this project.
3. Open https://share.streamlit.io/
4. Select the repository and branch.
5. Set the main file path to `app.py`.
6. Deploy.

No Hugging Face API token is required: the models are downloaded from their public Hugging Face repositories at runtime.

## Important deployment limitation

The translation model (`facebook/m2m100_418M`) and speech model need substantial memory and CPU time. Streamlit Community Cloud resources can be insufficient or slow, depending on the current plan and load. If the app repeatedly restarts or runs out of memory, use a machine with more RAM or host model inference separately. This is a real resource constraint, not a missing code file.

The app caches both models during a running session to avoid downloading/loading them for every recording. A cold restart may require loading them again.

## Accuracy and responsible use

This is a working prototype, not a certified interpreter. Accuracy varies by language pair, accent, speaking speed, microphone quality, and background noise. Test it with native speakers before presenting it as reliable. Do not use it for medical, legal, emergency, or other high-stakes communication.

The model cards and upstream licenses should be reviewed before commercial use:
- Translation: https://huggingface.co/facebook/m2m100_418M
- Speech recognition: https://huggingface.co/Systran/faster-whisper-base
- Streamlit microphone input: https://docs.streamlit.io/develop/api-reference/widgets/st.audio_input

## Troubleshooting

- **Microphone does not appear:** Open the deployed app in a browser that supports microphone access and allow microphone permission. Use HTTPS.
- **No speech detected:** Record a short, clear sentence in the selected source language.
- **Translation is slow:** This model runs on CPU. Shorter sentences are faster.
- **Out of memory / app restarts:** The hosting instance does not have enough RAM for the models. Use a larger instance or separate inference hosting.
- **Audio generation fails:** gTTS needs an internet connection.
