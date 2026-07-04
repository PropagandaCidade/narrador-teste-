import os, io, logging, re, base64, httpx, json
from flask import Flask, request, jsonify, send_file, make_response
from flask_cors import CORS
from pydub import AudioSegment, effects

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HiveWorker")

app = Flask(__name__)
CORS(app, expose_headers=['X-Model-Used', 'X-Prompt-Tokens', 'X-Output-Tokens'])

@app.route('/')
def home():
    return "Worker TTS Teste v1.0"

@app.route('/api/generate-audio', methods=['POST'])
def generate_audio_endpoint():
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "JSON invalido ou ausente"}), 400

        api_key = data.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return jsonify({"error": "Chave Gemini ausente"}), 500

        text = data.get('text', '').strip()
        voice = str(data.get('voice', 'Kore')).capitalize()
        model_nickname = str(data.get('model_to_use', 'flash')).lower()

        if not text or not voice:
            return jsonify({"error": "Texto e voz obrigatorios"}), 400

        # Usar sempre o texto puro, sem concatenar custom_prompt
        final_text = text

        if "3.1" in model_nickname:
            model_fullname = "gemini-3.1-flash-tts-preview"
        elif "pro" in model_nickname:
            model_fullname = "gemini-2.5-pro-preview-tts"
        else:
            model_fullname = "gemini-2.5-flash-preview-tts"

        logger.info(f"Modelo: {model_fullname} | Texto: {len(final_text)} chars")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_fullname}:generateContent?key={api_key}"

        payload = {
            "contents": [{"parts": [{"text": final_text}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {
                            "voice_name": voice
                        }
                    }
                }
            },
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "BLOCK_NONE"}
            ]
        }

        with httpx.Client(timeout=120.0) as client:
            res = client.post(url, json=payload)
            res_json = res.json()

            if res.status_code != 200:
                err = res_json.get('error', {})
                detail = err.get('message', json.dumps(err))
                logger.error(f"Gemini ERRO {res.status_code}: {detail}")
                return jsonify({
                    "error": detail,
                    "code": res.status_code,
                    "payload_enviado": payload
                }), res.status_code

            parts = res_json.get('candidates', [{}])[0].get('content', {}).get('parts', [])
            if not parts or 'inlineData' not in parts[0]:
                return jsonify({
                    "error": "Gemini nao retornou audio inlineData",
                    "resposta_completa": res_json
                }), 500

            audio_bytes = base64.b64decode(parts[0]['inlineData']['data'])

        audio = AudioSegment.from_raw(io.BytesIO(audio_bytes), sample_width=2, frame_rate=24000, channels=1)
        audio = audio.set_channels(1).set_sample_width(2)
        audio = effects.normalize(audio, headroom=0.45)
        audio = audio.set_frame_rate(44100).set_channels(1).set_sample_width(2)

        mp3 = io.BytesIO()
        audio.export(mp3, format="mp3", bitrate="192k", parameters=["-ac", "1", "-ar", "44100"])
        mp3.seek(0)

        resp = make_response(send_file(io.BytesIO(mp3.getvalue()), mimetype='audio/mpeg'))
        resp.headers['X-Model-Used'] = model_fullname
        return resp

    except Exception as e:
        logger.error(f"Erro no worker: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)