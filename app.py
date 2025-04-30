import gradio as gr
from transformers import MarianMTModel, MarianTokenizer

en_to_ur_model_name = "Helsinki-NLP/opus-mt-en-ur"
ur_to_en_model_name = "Helsinki-NLP/opus-mt-ur-en"

en_to_ur_tokenizer = MarianTokenizer.from_pretrained(en_to_ur_model_name)
en_to_ur_model = MarianMTModel.from_pretrained(en_to_ur_model_name)

ur_to_en_tokenizer = MarianTokenizer.from_pretrained(ur_to_en_model_name)
ur_to_en_model = MarianMTModel.from_pretrained(ur_to_en_model_name)

def translate(text, direction):
    if not text.strip():
        return "Please enter some text."
    
    if direction == "English → Urdu":
        tokenizer = en_to_ur_tokenizer
        model = en_to_ur_model
    else:
        tokenizer = ur_to_en_tokenizer
        model = ur_to_en_model

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**inputs)
    output = tokenizer.decode(translated[0], skip_special_tokens=True)
    return output
#Gradio usage here
with gr.Blocks() as demo:
    gr.Markdown("## 🌍 English ↔ Urdu Translator")
    gr.Markdown("Translate between English and Urdu using Hugging Face models!")

    with gr.Row():
        direction = gr.Radio(choices=["English → Urdu", "Urdu → English"], value="English → Urdu", label="Translation Direction")

    text_input = gr.Textbox(lines=4, label="Input Text")
    translate_btn = gr.Button("🔁 Translate")
    output_text = gr.Textbox(lines=4, label="Translated Text")

    translate_btn.click(fn=translate, inputs=[text_input, direction], outputs=output_text)

demo.launch()
