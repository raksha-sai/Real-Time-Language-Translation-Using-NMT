import sys
import os
from transformers import MarianMTModel, MarianTokenizer

# Suppress unnecessary logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Supported language codes
LANGUAGE_CODES = {
    "English": "en", "Spanish": "es", "French": "fr", "German": "de", "Italian": "it",
    "Dutch": "nl", "Russian": "ru", "Chinese": "zh", "Japanese": "ja", "Korean": "ko"
}

def load_model(src_lang, tgt_lang):
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    print(f"Loading model: {model_name}")
    try:
        model = MarianMTModel.from_pretrained(model_name)
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def translate_text(text, model, tokenizer):
    try:
        tokens = tokenizer.encode(text, return_tensors="pt", truncation=True)
        translated_tokens = model.generate(tokens, max_length=512, num_beams=4, early_stopping=True)
        return tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Translation error: {e}")
        return None

def main():
    print("Welcome to the Multi-Language Translator!")
    print("Available languages:")
    for lang, code in LANGUAGE_CODES.items():
        print(f"- {lang} ({code})")

    src_lang = input("Enter source language: ").strip()
    tgt_lang = input("Enter target language: ").strip()
    
    if src_lang not in LANGUAGE_CODES or tgt_lang not in LANGUAGE_CODES:
        print("Unsupported language. Please choose from the list.")
        return
    
    src_code = LANGUAGE_CODES[src_lang]
    tgt_code = LANGUAGE_CODES[tgt_lang]
    
    model, tokenizer = load_model(src_code, tgt_code)
    if model is None:
        print("Could not load translation model.")
        return

    while True:
        text = input(f"Enter text to translate from {src_lang} to {tgt_lang} (or type 'exit' to quit): ").strip()
        if text.lower() == 'exit':
            print("Goodbye!")
            break

        translated_text = translate_text(text, model, tokenizer)
        if translated_text:
            print(f"Translated Text: {translated_text}")
        else:
            print("Could not translate the text.")

if __name__ == "__main__":
    main()
