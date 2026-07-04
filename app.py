import os, io, logging, re, base64, httpx, json
from flask import Flask, request, jsonify, send_file, make_response
from flask_cors import CORS
from pydub import AudioSegment, effects

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HiveWorker")

app = Flask(__name__)
CORS(app, expose_headers=['X-Model-Used'])

FONEMAS_API_URL = os.environ.get(
    "FONEMAS_API_URL",
    "https://propagandacidadeaudio.com.br/voice-hub/admin/fonemas/api.php"
)

_rules_cache = None

def _load_rules():
    global _rules_cache
    if _rules_cache is not None:
        return _rules_cache
    try:
        resp = httpx.get(FONEMAS_API_URL, timeout=10.0)
        if resp.status_code == 200:
            data = resp.json()
            gold = data.get("gold_rules", {})
            pr_map = {}
            for key, rule in gold.items():
                if isinstance(rule, dict) and "replace" in rule:
                    searches = rule.get("search", [])
                    replacement = rule["replace"]
                    for s in searches:
                        if s:
                            pr_map[s] = replacement
                elif isinstance(rule, str):
                    pr_map[key] = rule
            _rules_cache = pr_map
            logger.info(f"Regras foneticas carregadas: {len(pr_map)} entradas")
            return _rules_cache
    except Exception as e:
        logger.warning(f"Falha ao carregar regras foneticas: {e}")
    return {}


def _apply_pronunciation_guide(text):
    rules = _load_rules()
    for word, replacement in rules.items():
        pattern = re.compile(re.escape(word), re.IGNORECASE)
        text = pattern.sub(replacement, text)
    return text


@app.route('/')
def home():
    return "Worker TTS v3.1 - Pronunciation Map Active"

def _voice_profile(voice_name):
    profiles = {
        "Kore": "Kore, locutora profissional de rádio, português brasileiro, tom claro e natural",
        "Aoede": "Aoede, locutor masculino, português brasileiro, voz firme e comercial",
        "Puck": "Puck, locutor jovem, português brasileiro, tom descontraído e energético",
        "Charon": "Charon, locutor masculino, português brasileiro, voz grave e imponente",
        "Fenrir": "Fenrir, locutor masculino, português brasileiro, tom sério e institucional",
    }
    return profiles.get(voice_name, f"{voice_name}, locutor, português brasileiro")


def _build_payload(text, voice, custom_prompt, mode, scene="", director_notes=""):
    if mode == 'pure':
        return {
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
        final_text = f"[CONTEXTO: {custom_prompt}] {text}" if custom_prompt else text
        return {
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
        final_text = f"{text}\n\n[CONTEXTO: {custom_prompt}]" if custom_prompt else text
        return {
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
        return payload

    elif mode == 'system_no_audio_config':
        return {
            "systemInstruction": {"parts": [{"text": custom_prompt}]},
            "contents": [{"parts": [{"text": text}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"]
            }
        }

    elif mode == 'chirp':
        return {
            "contents": [{"parts": [{"text": f"[ESTILO: CHIRP HD] {text}"}]}],
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
        final_text = f"{custom_prompt}\n\n{text}" if custom_prompt else text
        return {
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
        return {
            "contents": [{"parts": [{"text": f"Narre o texto a seguir: {text}"}]}],
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
        return {
            "contents": [{"parts": [{"text": f"<<INICIO DA NARRACAO>> {text} <<FIM DA NARRACAO>>"}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {"voice_name": voice}
                    }
                }
            }
        }

    elif mode == 'voice_director':
        profile = _voice_profile(voice)
        sc = scene or "Narração de texto publicitário"
        notes = director_notes or (
            "Leia o TRANSCRIPT exatamente como escrito, palavra por palavra. "
            "Preserve siglas, marcas, nomes próprios e números estrangeiros. "
            "Não interprete, não reformule e não autocorrija siglas, marcas ou palavras escritas em maiúsculas."
        )
        final_text = (
            f"AUDIO PROFILE\n{profile}\n\n"
            f"THE SCENE\n{sc}\n\n"
            f"DIRECTOR'S NOTES\n{notes}\n\n"
            f"TRANSCRIPT\n{text}"
        )
        return {
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
        return None


@app.route('/api/generate-audio', methods=['POST'])
def generate_audio_endpoint():
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "JSON invalido"}), 400

        text = data.get('text', '').strip()
        voice = str(data.get('voice', 'Kore')).capitalize()
        custom_prompt = data.get('custom_prompt', '').strip()
        mode = data.get('mode', 'pure').strip()
        model_nickname = str(data.get('model_to_use', 'flash')).lower()
        debug = data.get('debug', False)
        scene = data.get('scene', '').strip()
        director_notes = data.get('director_notes', '').strip()

        if not text or not voice:
            return jsonify({"error": "Texto e voz obrigatorios"}), 400

        if "3.1" in model_nickname:
            model_fullname = "gemini-3.1-flash-tts-preview"
        elif "pro" in model_nickname:
            model_fullname = "gemini-2.5-pro-preview-tts"
        else:
            model_fullname = "gemini-2.5-flash-preview-tts"

        use_phonetic = data.get('phonetic', True)
        if use_phonetic:
            text = _apply_pronunciation_guide(text)

        # --- MONTAGEM DO PAYLOAD CONFORME MODO ---
        payload = _build_payload(text, voice, custom_prompt, mode, scene, director_notes)
        if not payload:
            return jsonify({"error": f"Modo desconhecido: {mode}", "modos_disponiveis": ["pure","inline","inline_suffix","system","system_no_audio_config","chirp","voice_instruction","prefixed_narrate","delimiters","voice_director"]}), 400

        safety = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "BLOCK_NONE"}
        ]
        payload["safetySettings"] = safety

        if debug:
            payload_compact = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            curl_cmd = f'curl -s -X POST "https://generativelanguage.googleapis.com/v1beta/models/{model_fullname}:generateContent?key=SUA_CHAVE_AQUI" -H "Content-Type: application/json" -d \'{payload_compact}\''
            return jsonify({
                "debug": True,
                "mode": mode,
                "model": model_fullname,
                "voice": voice,
                "payload": payload,
                "curl": curl_cmd,
                "instrucao": "Copie o comando 'curl' acima, troque SUA_CHAVE_AQUI pela sua API key do Gemini, e execute no terminal. Compare o resultado com o que o worker retorna."
            })

        api_key = data.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return jsonify({"error": "Chave Gemini ausente"}), 500

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_fullname}:generateContent?key={api_key}"

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