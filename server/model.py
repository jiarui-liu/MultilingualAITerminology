import json
from nltk import word_tokenize, sent_tokenize
from dataclasses import dataclass
from enum import Enum
from server.acl_antho_translator.prompt_refine import AnthoPromptRefine
import re
from nltk.tokenize import sent_tokenize

def split_paragraph(paragraph, max_words=64):
    # Step 1: Split the paragraph into sentences
    sentences = sent_tokenize(paragraph)
    
    # Step 2: Handle newline characters within sentences
    split_sentences = []
    for sentence in sentences:
        sub_sentences = re.split(r'(\n+)', sentence)
        split_sentences.extend(sub_sentences)
    
    chunks = []
    current_chunk = []

    def add_sentence_to_chunk(sentence, chunk):
        words = word_tokenize(sentence)
        if len(words) > max_words:
            # If the sentence itself is longer than max_words, split it further
            sub_chunks = split_long_sentence(sentence, max_words)
            chunk.extend(sub_chunks)
        else:
            chunk.append(sentence)
        return chunk

    def split_long_sentence(sentence, max_words):
        
        words = word_tokenize(sentence)
        sub_chunks = []
        current_sub_chunk = []
        for word in words:
            current_sub_chunk.append(word)
            if len(current_sub_chunk) >= max_words:
                sub_chunks.append(' '.join(current_sub_chunk))
                current_sub_chunk = []
        if current_sub_chunk:
            sub_chunks.append(' '.join(current_sub_chunk))
        return sub_chunks

    # Step 3: Join sentences into chunks
    for sentence in split_sentences:
        current_chunk = add_sentence_to_chunk(sentence, current_chunk)
        current_chunk_str = ' '.join(current_chunk)
        if len(word_tokenize(current_chunk_str)) > max_words:
            # Remove the last added sentence to keep the chunk under max_words
            removed_sentence = current_chunk.pop()
            chunks.append(' '.join(current_chunk))
            current_chunk = [removed_sentence]

    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


class Translator:
    def __init__(self, args):
        self.args = args
        
        self.tgt_langs = [
            "Chinese",
            "Arabic",
            "French",
            "Russian",
            "Japanese"
        ]
        
        self.set_model()
        from translator.refiner import TermAwareRefiner
        class RefinerArgs:
            model = self.args.model_refine
            openai_key_path = self.args.openai_key_path
        refiner_args = RefinerArgs
        self.refiner = TermAwareRefiner(refiner_args, self.args.term_path)
    
    def set_model(self):
        if self.args.model_mt == "seamless":
            from translator.translator import SeamlessTranslator
            class Config:
                model_name = "facebook/hf-seamless-m4t-Large"
                cache_dir = self.args.cache_dir
                lang_dict = {
                    "Arabic": "arb",
                    "Chinese": "cmn",
                    "English": "eng",
                    "French": "fra",
                    "Japanese": "jpn",
                    "Russian": "rus",
                }
            config = Config
            self.model = SeamlessTranslator(config)
        elif self.args.model_mt == "googletrans":
            from translator.translator import GoogleTranslator
            from typing import Any
            import httpcore
            setattr(httpcore, 'SyncHTTPTransport', Any)
            class Config:
                lang_dict = {
                    "Arabic": "ar",
                    "Chinese": "zh-cn",
                    "English": "en",
                    "French": "fr",
                    "Japanese": "ja",
                    "Russian": "ru",
                }
            config = Config
            self.model = GoogleTranslator(config)
        
    # used in the server to translate a text directly or using prompt refinement
    def translate(self, text, src_lang, tgt_lang, mode, seamless=""):
        
        # if the default translation is provided and the mode is set to use prompt refinement
        if seamless != "" and mode == "term_aware":
            refiner = AnthoPromptRefine("gpt-4o-mini", "./data/")
            return refiner.translate(text, seamless, src_lang, tgt_lang)
        else:
            # if the default translation is not provided, translate first
            splitTxt = split_paragraph(text)
            result = ""
            for txt in splitTxt:
                result += " " + self.model.translate(txt, src_lang, tgt_lang)
            
            # then return if "direct" or run the prompt refinement if not
            if mode == "direct":
                return result.strip()
            elif mode == "term_aware":
                refiner = AnthoPromptRefine("gpt-4o-mini", "./data/")
                return refiner.translate(text, result.strip(), src_lang, tgt_lang)
        