# app.py - VERSÃO 29.4 - WORKER ENGINE (HIVE STABLE) - CHIRP & 3.1 SUPPORT
# LOCAL: Repositório Único (N1, N2, N3, N4, N5) no Railway
# DESCRIÇÃO: Identifica corretamente Chirp e Gemini 3.1 para o Analytics Global.
# VERSÃO: 29.4 - FIXED SAFE: MP3 mono, 44.100 Hz, 192 kbps, base 16-bit, sem codec explícito para evitar crash

import os
import io
import logging
import re
import time
import base64
import httpx
import json

from flask import Flask, request, jsonify, send_file, make_response
from flask_cors import CORS
from pydub import AudioSegment, effects

# 1. CONFIGURAÇÃO DE LOGS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HiveWorker")

app = Flask(__name__)
CORS(app, expose_headers=['X-Model-Used', 'X-Prompt-Tokens', 'X-Output-Tokens'])

def clean_skill_tags(text):
    if not text:
        return ""
    cleaned = re.sub(r'</?context_guard>', '', text)
    return cleaned.strip()

@app.route('/')
def home():
    srv = os.environ.get('RAILWAY_SERVICE_NAME', 'Worker')
    return f"Serviço v29.4 ({srv}) - Tier 2 Analytics Ready."

@app.route('/api/generate-audio', methods=['POST'])
def generate_audio_endpoint():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Dados inválidos."}), 400

        api_key = data.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return jsonify({"error": "Chave Gemini ausente."}), 500

        text_raw = data.get('text', '')
        text_to_narrate = clean_skill_tags(text_raw)
        voice_name = str(data.get('voice', 'Kore')).capitalize()
        model_nickname = str(data.get('model_to_use', 'flash')).lower()
        custom_prompt = data.get('custom_prompt', '').strip()
        origin = data.get('origin_interface', 'dashboard')
        
        try:
            temperature = float(data.get('temperature', 0.85))
        except:
            temperature = 0.85

        if not text_to_narrate or not voice_name:
            return jsonify({"error": "Texto e voz obrigatórios."}), 400

        # --- LÓGICA DE SELEÇÃO DE MODELO E IDENTIFICAÇÃO (PARA HEADERS) ---
        if "chirp" in model_nickname:
            # Se for Chirp, usamos o Flash 2.5 como motor (ou o motor que você configurou para Chirp)
            # Mas a etiqueta X-Model-Used será 'chirp' para o Analytics cobrar US$ 30/1M chars
            final_text_for_api = f"[ESTILO: CHIRP HD] {text_to_narrate}"
            model_fullname = "gemini-2.5-flash-preview-tts" 
            analytics_label = "chirp"
            
        elif "3.1" in model_nickname:
            if custom_prompt:
                final_text_for_api = f"Instrução de narração: {custom_prompt}\n\nTexto para narrar: {text_to_narrate}"
            else:
                final_text_for_api = text_to_narrate
            model_fullname = "gemini-3.1-flash-tts-preview"
            analytics_label = model_fullname
            
        else:
            final_text_for_api = f"[CONTEXTO: {custom_prompt}] {text_to_narrate}" if custom_prompt else text_to_narrate
            if "pro" in model_nickname:
                model_fullname = "gemini-2.5-pro-preview-tts"
            else:
                model_fullname = "gemini-2.5-flash-preview-tts"
            analytics_label = model_fullname

        logger.info(f"HIVE Worker: {origin} -> {analytics_label}")

        # --- CHAMADA REST ---
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_fullname}:generateContent?key={api_key}"
        
        voice_seed = int(data.get('voice_seed', 0)) or abs(hash(voice_name)) % 100000
        payload = {
            "contents": [{"parts": [{"text": final_text_for_api}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "seed": voice_seed,
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {
                            "voice_name": voice_name
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

        response_audio_bytes = None
        prompt_tokens = 0
        output_tokens = 0

        with httpx.Client(timeout=120.0) as client:
            res = client.post(url, json=payload)
            res_json = res.json()
            
            if res.status_code == 200:
                usage = res_json.get('usageMetadata', {})
                prompt_tokens = usage.get('promptTokenCount', 0)
                output_tokens = usage.get('candidatesTokenCount', 0)

                if 'candidates' in res_json and len(res_json['candidates']) > 0:
                    parts = res_json['candidates'][0].get('content', {}).get('parts', [])
                    if parts and 'inlineData' in parts[0]:
                        response_audio_bytes = base64.b64decode(parts[0]['inlineData']['data'])
            else:
                err_msg = res_json.get('error', {}).get('message', f"API Error HTTP {res.status_code}")
                logger.error(f"Gemini FAIL | model={model_fullname} | voice={voice_name} | text_len={len(text_to_narrate)} | status={res.status_code} | err={err_msg}")
                return jsonify({"error": err_msg}), res.status_code

        if not response_audio_bytes:
            return jsonify({"error": "Falha na geração."}), 500

        # --- PROCESSAMENTO DE ÁUDIO ---
        # O áudio bruto da API deve ser lido na taxa original correta.
        # Não use 44.100 Hz diretamente no from_raw, pois isso acelera a voz e deixa fina.
        audio_segment = AudioSegment.from_raw(
            io.BytesIO(response_audio_bytes),
            sample_width=2,   # 16-bit PCM
            frame_rate=24000, # taxa original do áudio bruto retornado pela API
            channels=1        # voz mono
        )

        # Mantém a voz em mono e base 16-bit.
        audio_segment = audio_segment.set_channels(1)
        audio_segment = audio_segment.set_sample_width(2)

        # Normaliza volume sem mexer em velocidade/pitch.
        audio_segment = effects.normalize(audio_segment, headroom=0.45)

        # Converte para 44.100 Hz sem causar efeito "voz de esquilo".
        audio_segment = audio_segment.set_frame_rate(44100)

        # Reforça mono e 16-bit depois da conversão.
        audio_segment = audio_segment.set_channels(1)
        audio_segment = audio_segment.set_sample_width(2)

        mp3_buffer = io.BytesIO()

        # Exportação segura:
        # Remove codec explícito e bitrate duplicado para evitar crash em ambientes Railway/FFmpeg.
        # O parâmetro -ac 1 força a saída final em mono.
        # O parâmetro -ar 44100 força a saída final em 44.100 Hz.
        audio_segment.export(
            mp3_buffer,
            format="mp3",
            bitrate="192k",
            parameters=[
                "-ac", "1",
                "-ar", "44100"
            ]
        )
        mp3_buffer.seek(0)
        
        http_response = make_response(send_file(io.BytesIO(mp3_buffer.getvalue()), mimetype='audio/mpeg'))
        
        # HEADERS DE TELEMETRIA
        http_response.headers['X-Model-Used'] = analytics_label
        http_response.headers['X-Prompt-Tokens'] = str(prompt_tokens)
        http_response.headers['X-Output-Tokens'] = str(output_tokens)
        
        return http_response

    except Exception as e:
        logger.error(f"Erro: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
