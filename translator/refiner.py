import os
import json
import pandas as pd
from fuzzywuzzy import fuzz
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from translator.translator import Translator


class TermAwareRefiner:
    def __init__(self, args, term_path):
    
        self.terms = {}
        for lang in ["Chinese", "Arabic", "French", "Russian", "Japanese"]:
            self.terms[lang] = pd.read_csv(os.path.join(term_path, lang + ".csv")).to_dict(orient='records')
        self.args = args
        
    def find_terminology(self, paragraph, terminology):
        def preprocess_text(text):
            lemmatizer = WordNetLemmatizer()
            tokens = word_tokenize(text.lower())
            return ' '.join([lemmatizer.lemmatize(token) for token in tokens])
        preprocessed_paragraph = preprocess_text(paragraph)
        preprocessed_terminology = preprocess_text(terminology)
        return fuzz.partial_ratio(preprocessed_terminology, preprocessed_paragraph) > 80  # Adjust threshold as needed
        

    def get_related_term_lst(self, result, tgt_lang):
        terms_dict = {}
        for term in self.terms[tgt_lang]:
            if self.find_terminology(result, term['English']):
                terms_dict[term['English']] = term[tgt_lang]
        return terms_dict
    
    def format_term_str(self, term_dict):
        return "- " + "- ".join([f"{key}: {val}" for key, val in term_dict.items()])
    
    def refine_translation(self, result, text, src_lang, tgt_lang):
        from translator.prompt_utils import prompt_refine
        from translator.gpt_utils import gpt_completion
        term_lst = self.get_related_term_lst(result, tgt_lang)
        term_str = self.format_term_str(term_lst)
        prompt = prompt_refine.format(
            text=text,
            result=result,
            term_str=term_str,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
        )
        return gpt_completion(
            prompt = prompt,
            model=self.args.model,
            temperature = 0,
            max_tokens = 1024,
            openai_key_path = self.args.openai_key_path
        )

