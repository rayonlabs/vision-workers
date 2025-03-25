import gradio as gr
import requests
import base64
from io import BytesIO
import argparse
import tempfile
import os

# Set up command line argument parsing
parser = argparse.ArgumentParser(description='Run Gradio app for 2-speaker voice cloning.')
parser.add_argument('--url', type=str, help='API endpoint URL', required=True)
parser.add_argument('--port', type=int, help='Port number on which the app will run', default=7860)
args = parser.parse_args()

def encode_audio_to_base64(audio_path):
    with open(audio_path, "rb") as audio_file:
        audio_content = audio_file.read()
    base64_audio = base64.b64encode(audio_content).decode("utf-8")
    return f"data:audio/wav;base64,{base64_audio}"

def call_api(caption_a, audio_a, caption_b, audio_b, conversation, seed):
    # Convert uploaded files to base64 strings
    try:
        audio_b64_a = encode_audio_to_base64(audio_a)
        audio_b64_b = encode_audio_to_base64(audio_b)

        data = {
            "text_prompt_a": caption_a,
            "audio_prompt_a": audio_b64_a,
            "text_prompt_b": caption_b,
            "audio_prompt_b": audio_b64_b,
            "conversation": conversation,
            "seed": int(seed),
            "max_audio_length_ms": 30000
        }

        response = requests.post(args.url, json=data)
        response_data = response.json()

        output_audio_b64 = response_data["audio_b64"]
        output_audio_bytes = base64.b64decode(output_audio_b64)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", mode="wb") as tmp_file:
            tmp_file.write(output_audio_bytes)
            return tmp_file.name
    except Exception as e:
        print(f"Error during API call: {e}")
        return None

with gr.Blocks() as app:
    gr.Markdown("# 🗣️ Two-Speaker Voice Cloning\nProvide two voice samples and captions. Enter a multi-turn conversation with alternating lines.")
    
    with gr.Row():
        with gr.Column():
            caption_a = gr.Textbox(label="Caption for Speaker A", value="This is speaker A.")
            audio_a = gr.Audio(label="Speaker A Audio (.wav, ~6s)", type="filepath")
        
        with gr.Column():
            caption_b = gr.Textbox(label="Caption for Speaker B", value="This is speaker B.")
            audio_b = gr.Audio(label="Speaker B Audio (.wav, ~6s)", type="filepath")
    
    conversation_input = gr.Textbox(
        label="Conversation (one line per speaker turn)",
        lines=6,
        value="Hey how are you doing.\nPretty good, pretty good.\nI'm great, so happy to be speaking to you."
    )
    
    seed_input = gr.Number(value=0, label="Seed (integer)")
    
    submit_btn = gr.Button("Generate Conversation", variant="primary")
    output_audio = gr.Audio(label="Generated Conversation", interactive=False)

    submit_btn.click(
        fn=call_api,
        inputs=[caption_a, audio_a, caption_b, audio_b, conversation_input, seed_input],
        outputs=[output_audio]
    )

app.launch(server_port=args.port, share=True)
