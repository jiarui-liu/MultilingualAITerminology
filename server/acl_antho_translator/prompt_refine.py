import time
import re
import json
# import pandas as pd
import openai
from tqdm import tqdm
import sys
import argparse
from translator.get_terms import TermCollector
import os

# to set up the openai model 
def openai_setup(key_path=''):
    if (key_path == ""):
        key = os.environ["openai_api_key"]
    else:
        with open(key_path) as f:
            key = f.read().strip().split("\n")[0]
    
    print("Read key from", key_path)
    openai.api_key = key.strip()

# to prompt openai model to translate using prompt refinement 
def openai_prompt(src_text, tgt_text, relevant_terms_dict, tgt_lang, model='gpt-4o'):
    src_lang = 'English'
    
    tmp = []
    for src_term, tgt_term in relevant_terms_dict.items():
        if tgt_lang in tgt_term:
            tmp.append(f"{src_lang}: {src_term}, {tgt_lang}: {tgt_term[tgt_lang]}")
    
    if len(tmp) == 0:
        return None
    terms = "- ".join(tmp)
    
    prompt = f"""For the following translation into {tgt_lang}, please use the specified {tgt_lang} terms for the corresponding {src_lang} terms, while keeping the other content unchanged.

        Term dictionary:
        {terms}

        {src_lang} text:
        {src_text}

        {tgt_lang} translation:
        {tgt_text}

        If multiple terms are nested or overlap with the context in {src_lang}, select the longest span that matches the context. Additionally, if a term has multiple meanings, only replace the term if its original context is relevant to the AI field. Provided the updated translation only.
     """
     
    print(prompt)
    while True:
        try:
            resp = openai.ChatCompletion.create(
                model = model,
                messages = [{"role": "user", "content": prompt}],
                temperature = 0,
                max_tokens = 1024
            ).choices[0].message.content
            break
        except Exception as e:
            print(e)
        time.sleep(5)
    
    # extract valid_term, explanation
    return resp

class AnthoPromptRefine():
    def __init__(self, model, term_file_path):
        openai_setup()
        self.term_file_path = term_file_path
        self.model = model
        
    def translate(self, src_text, seamless_trans, src_lang, tgt_lang):
        self.src_text = src_text
        self.seamless_trans = seamless_trans
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        
        term_collector = TermCollector(self.term_file_path, [self.tgt_lang])
        
        # extract the relevant terms in the dictionary 
        relevant_terms_dict = {}
        for key in term_collector.find_terminology(self.src_text):
            relevant_terms_dict[key] = term_collector.terms_dict[key] 
            
        # translate using prompting
        translation = openai_prompt(
            self.src_text,
            self.seamless_trans,
            relevant_terms_dict = relevant_terms_dict,
            tgt_lang = self.tgt_lang,
            model=self.model
        )
        
        if translation == None:
            return self.seamless_trans
        return translation        