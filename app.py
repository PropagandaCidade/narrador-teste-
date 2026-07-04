import os, io, logging, re, base64, httpx, json
from flask import Flask, request, jsonify, send_file, make_response
from flask_cors import CORS
from pydub import AudioSegment, effects

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HiveWorker")

app = Flask(__name__)
CORS(app, expose_headers=['X-Model-Used'])

@app.route('/')
def home():
    return "Worker TTS v3.0 - Test Matrix"

@app.route('/api/generate-audio', methods=['POST'])
def generate_audio_endpoint():
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "JSON invalido"}), 400

        api_key = data.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return jsonify({"error": "Chave Gemini ausente"}), 500

        text = data.get('text', '').strip()
        voice = str(data.get('voice', 'Kore')).capitalize()
        custom_prompt = data.get('custom_prompt', '').strip()
        mode = data.get('mode', 'pure').strip()
        model_nickname = str(data.get('model_to_use', 'flash')).lower()

        if not text or not voice:
            return jsonify({"error": "Texto e voz obrigatorios"}), 400

        if "3.1" in model_nickname:
            model_fullname = "gemini-3.1-flash-tts-preview"
        elif "pro" in model_nickname:
            model_fullname = "gemini-2.5-pro-preview-tts"
        else:
            model_fullname = "gemini-2.5-flash-preview-tts"

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_fullname}:generateContent?key={api_key}"

        # --- MONTAGEM DO PAYLOAD CONFORME MODO ---
        payload = None

        if mode == 'pure':
            # MODO 1: Texto puro, sem instrucao alguma
            logger.info(f"MODO pure: texto puro, sem instrucao")
            payload = {
                "contents": [{"parts": [{"text": text}]}],
                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                    "speechConfig": {
                        "voiceConfig": {
                            "prebuiltVoiceConfig": {"voice_name": voice}
                        }
                    }
                }
            }

        elif mode == 'inline':
            # MODO 2: Instrucao DENTRO do texto (como era antes)
            logger.info(f"MODO inline: instrucao concatenada no texto")
            final_text = f"[CONTEXTO: {custom_prompt}] {text}" if custom_prompt else text
            payload = {
                "contents": [{"parts": [{"text": final_text}]}],
                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                    "speechConfig": {
                        "voiceConfig": {
                            "prebuiltVoiceConfig": {"voice_name": voice}
                        }
                    }
                }
            }

        elif mode == 'inline_suffix':
            # MODO 3: Instrucao DEPOIS do texto
            logger.info(f"MODO inline_suffix: instrucao depois do texto")
            final_text = f"{text}\n\n[CONTEXTO: {custom_prompt}]" if custom_prompt else text
            payload = {
                "contents": [{"parts": [{"text": final_text}]}],
                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                    "speechConfig": {
                        "voiceConfig": {
                            "prebuiltVoiceConfig": {"voice_name": voice}
                        }
                    }
                }
            }

        elif mode == 'system':
            # MODO 4: systemInstruction separada (testar se funciona em algum modelo)
            logger.info(f"MODO system: tentando systemInstruction")
            payload = {
                "contents": [{"parts": [{"text": text}]}],
                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                    "speechConfig": {
                        "voiceConfig": {
                            "prebuiltVoiceConfig": {"voice_name": voice}
                        }
                    }
                }
            }
            if custom_prompt:
                payload["systemInstruction"] = {"parts": [{"text": custom_prompt}]}

        elif mode == 'system_no_audio_config':
            # MODO 5: systemInstruction SEM speechConfig (alguns modelos precisam)
            logger.info(f"MODO system_no_audio_config: systemInstruction sem speechConfig")
            payload = {
                "systemInstruction": {"parts": [{"text": custom_prompt}]},
                "contents": [{"parts": [{"text": text}]}],
                "generationConfig": {
                    "responseModalities": ["AUDIO"]
                }
            }

        elif mode == 'chirp':
            # MODO 6: Estilo CHIRP (prefixo no texto)
            logger.info(f"MODO chirp: prefixo estilo CHIRP")
            final_text = f"[ESTILO: CHIRP HD] {text}"
            payload = {
                "contents": [{"parts": [{"text": final_text}]}],
                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                    "speechConfig": {
                        "voiceConfig": {
                            "prebuiltVoiceConfig": {"voice_name": voice}
                        }
                    }
                }
            }

        elif mode == 'voice_instruction':
            # MODO 7: Instrucao de voz DENTRO do texto como frase natural
            logger.info(f"MODO voice_instruction: instrucao como frase natural")
            final_text = f"{custom_prompt}\n\n{text}" if custom_prompt else text
            payload = {
                "contents": [{"parts": [{"text": final_text}]}],
                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                    "speechConfig": {
                        "voiceConfig": {
                            "prebuiltVoiceConfig": {"voice_name": voice}
                        }
                    }
                }
            }

        elif mode == 'prefixed_narrate':
            # MODO 8: Prefixo "Narre o texto a seguir:" antes do texto
            logger.info(f"MODO prefixed_narrate: Narre o texto a seguir")
            final_text = f"Narre o texto a seguir: {text}"
            payload = {
                "contents": [{"parts": [{"text": final_text}]}],
                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                    "speechConfig": {
                        "voiceConfig": {
                            "prebuiltVoiceConfig": {"voice_name": voice}
                        }
                    }
                }
            }

        elif mode == 'delimiters':
            # MODO 9: Texto entre delimitadores
            logger.info(f"MODO delimiters: texto entre aspas/tags")
            final_text = f"<<INICIO DA NARRACAO>> {text} <<FIM DA NARRACAO>>"
            payload = {
                "contents": [{"parts": [{"text": final_text}]}],
                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                    "speechConfig": {
                        "voiceConfig": {
                            "prebuiltVoiceConfig": {"voice_name": voice}
                        }
                    }
                }
            }

        else:
            return jsonify({"error": f"Modo desconhecido: {mode}", "modos_disponiveis": ["pure","inline","inline_suffix","system","system_no_audio_config","chirp","voice_instruction","prefixed_narrate","delimiters"]}), 400

        payload["safetySettings"] = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "BLOCK_NONE"}
        ]

        logger.info(f"Modo={mode} | Modelo={model_fullname} | Voice={voice}")

        # --- CHAMADA GEMINI ---
        with httpx.Client(timeout=120.0) as client:
            res = client.post(url, json=payload)
            res_json = res.json()

            if res.status_code != 200:
                err = res_json.get('error', {})
                detail = err.get('message', json.dumps(err))
                logger.error(f"Gemini ERRO {res.status_code}: {detail}")
                # Retorna info completa para debug
                return jsonify({
                    "mode": mode,
                    "status": res.status_code,
                    "error": detail,
                    "payload_preview": {k: v for k, v in payload.items() if k != "contents" or True}
                }), res.status_code

            parts = res_json.get('candidates', [{}])[0].get('content', {}).get('parts', [])
            if not parts or 'inlineData' not in parts[0]:
                return jsonify({
                    "mode": mode,
                    "error": "Gemini nao retornou audio",
                    "resposta": res_json.get('candidates', [{}])[0].get('finishReason', 'unknown'),
                    "payload_preview": {k: v for k, v in payload.items() if k != "contents" or True}
                }), 500

            audio_bytes = base64.b64decode(parts[0]['inlineData']['data'])

        audio = AudioSegment.from_raw(io.BytesIO(audio_bytes), sample_width=2, frame_rate=24000, channels=1)
        audio = audio.set_channels(1).set_sample_width(2).set_frame_rate(44100)
        audio = effects.normalize(audio, headroom=0.45)

        mp3 = io.BytesIO()
        audio.export(mp3, format="mp3", bitrate="192k", parameters=["-ac", "1", "-ar", "44100"])
        mp3.seek(0)

        resp = make_response(send_file(io.BytesIO(mp3.getvalue()), mimetype='audio/mpeg'))
        resp.headers['X-Mode'] = mode
        resp.headers['X-Model-Used'] = model_fullname
        return resp

    except Exception as e:
        logger.error(f"Erro: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)