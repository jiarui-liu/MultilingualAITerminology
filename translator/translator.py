import re
import time
import transformers
import torch
from transformers import PhrasalConstraint
from transformers import LogitsProcessor

class Translator:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.prepare_model()
        
    def prepare_model(self):
        """Prepare necessary configurations of using the translator model."""
        pass
    
    def translate(self):
        """Translate text from source language to target language."""
        pass


class TerminologyAwareLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, en_text, lang, soft_penalty, lang_dict, tokens):
        self.tokenizer = tokenizer
        self.en_text = en_text
        self.lang = lang
        self.soft_penalty = soft_penalty
        self.lang_dict = lang_dict
        self.tokens = tokens

    def __call__(self, input_ids, scores):
        # print(scores)
        for idx in range(scores.shape[-1]):
            tokens = list({item for sublist in self.tokens for item in sublist})
            if idx in tokens:  # Penalize tokens not in the translation
                scores[:, idx] /= self.soft_penalty  # Apply soft penalty
        # print(scores)
        return scores


class NLLBTranslator(Translator):
    """
    - model_name: "facebook/nllb-200-3.3B"
    """
    
    def prepare_model(self):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name, cache_dir=self.args.cache_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.args.model_name, cache_dir=self.args.cache_dir).to(self.device)
        
    def translate(self, text, src_lang, tgt_lang):
        self.tokenizer.src_lang = self.args.lang_dict[src_lang]
        encoded_input = self.tokenizer(
            text,
            return_tensors="pt"
        )
        output_tokens = self.model.generate(
            **encoded_input.to(self.device),
            forced_bos_token_id = self.tokenizer.lang_code_to_id[self.args.lang_dict[tgt_lang]],
        )
        translated_text = self.tokenizer.batch_decode(
            output_tokens,
            skip_special_tokens=True
        )[0]
        return translated_text
    
    def translate_cbs(self, text, src_lang, tgt_lang, force_words):
        self.tokenizer.src_lang = self.args.lang_dict[src_lang]
        encoded_input = self.tokenizer(
            text,
            return_tensors="pt"
        )
        
        print(force_words)
        constraints = [
            PhrasalConstraint(self.tokenizer(
                force_word,
                add_special_tokens=False
            ).input_ids) for force_word in force_words
        ]
        if len(constraints) == 0:
            constraints = None
            
        output_tokens = self.model.generate(
            **encoded_input.to(self.device),
            forced_bos_token_id = self.tokenizer.lang_code_to_id[self.args.lang_dict[tgt_lang]],
            constraints=constraints,
            num_beams=5,
            num_return_sequences=1,
            no_repeat_ngram_size=1,
            remove_invalid_values=True,
        )
        translated_text = self.tokenizer.decode(
            output_tokens[0],
            skip_special_tokens=True
        )
        return translated_text

    def translate_constraint_soft(self, text, src_lang, tgt_lang, terms_dict, soft_penalty=0.8):
        self.tokenizer.src_lang = self.args.lang_dict[src_lang]
        encoded_input = self.tokenizer(
            text,
            return_tensors="pt"
        )
        
        tokens = []
        for translation in terms_dict:
            token = self.tokenizer(translation)['input_ids']
            tokens.append(token)
        
        logits_processor = TerminologyAwareLogitsProcessor(
            tokenizer=self.tokenizer,
            en_text=text,
            lang=tgt_lang,
            soft_penalty=soft_penalty,
            lang_dict=self.args.lang_dict,
            tokens = tokens,
        )
        
        output_tokens = self.model.generate(
            **encoded_input.to(self.device),
            forced_bos_token_id = self.tokenizer.lang_code_to_id[self.args.lang_dict[tgt_lang]],
            logits_processor = [logits_processor],
        )
        translated_text = self.tokenizer.batch_decode(
            output_tokens,
            skip_special_tokens=True
        )[0]
        return translated_text
        
    
class SeamlessTranslator(Translator):
    """
    args: an instance of M4TLarge from config.py.
    - model_name: "facebook/hf-seamless-m4t-Large"
    - cache_dir: "/data/user_data/jiaruil5/.cache/"
    
    """
    def prepare_model(self):
        from transformers import AutoProcessor
        from translator.transformers_customized.models.seamless_m4t.modeling_seamless_m4t import SeamlessM4TModel
        self.processor = AutoProcessor.from_pretrained(self.args.model_name, cache_dir=self.args.cache_dir, use_fast=False)
        self.model = SeamlessM4TModel.from_pretrained(self.args.model_name, cache_dir=self.args.cache_dir).to(self.device)

    def translate(self, text, src_lang, tgt_lang):
        text_inputs = self.processor(
            text = text,
            src_lang = self.args.lang_dict[src_lang],
            return_tensors="pt"
        )
        output_tokens = self.model.generate(
            **text_inputs.to(self.device),
            tgt_lang=self.args.lang_dict[tgt_lang],
            generate_speech=False
        )
        translated_text = self.processor.decode(
            output_tokens[0].tolist()[0],
            skip_special_tokens=True
        )
        return translated_text
    
    def translate_cbs(self, text, src_lang, tgt_lang, force_words):
        text_inputs = self.processor(
            text = text,
            src_lang = self.args.lang_dict[src_lang],
            return_tensors="pt"
        )
        
        print(force_words)
        constraints = [
            PhrasalConstraint(self.processor(
                force_word,
                src_lang = self.args.lang_dict[tgt_lang],
                # add_special_tokens=False
            ).input_ids) for force_word in force_words
        ]
        if len(constraints) == 0:
            constraints = None
        
        output_tokens = self.model.generate(
            **text_inputs.to(self.device),
            tgt_lang=self.args.lang_dict[tgt_lang],
            generate_speech=False,
            constraints=constraints,
            num_beams=10,
            num_return_sequences=1,
            no_repeat_ngram_size=1,
            remove_invalid_values=True,
        )
        translated_text = self.processor.decode(
            output_tokens[0].tolist()[0],
            skip_special_tokens=True
        )
        return translated_text

    def translate_constraint_soft(self, text, src_lang, tgt_lang, terms_dict, soft_penalty=0.8):
        text_inputs = self.processor(
            text = text,
            src_lang = self.args.lang_dict[src_lang],
            return_tensors="pt"
        )
        
        tokens = []
        for translation in terms_dict:
            token = self.processor(
                translation,
                src_lang=self.args.lang_dict[tgt_lang]
            )['input_ids']
            tokens.append(token)
        
        logits_processor = TerminologyAwareLogitsProcessor(
            tokenizer=self.processor,
            en_text=text,
            lang=tgt_lang,
            soft_penalty=soft_penalty,
            lang_dict=self.args.lang_dict,
            tokens = tokens,
        )
        
        output_tokens = self.model.generate(
            **text_inputs.to(self.device),
            tgt_lang=self.args.lang_dict[tgt_lang],
            generate_speech=False,
            logits_processor=[logits_processor]
        )
        translated_text = self.processor.decode(
            output_tokens[0].tolist()[0],
            skip_special_tokens=True
        )
        return translated_text

class AyaTranslator(Translator):
    """
    - model_name: "CohereForAI/aya-expanse-8b"
    """
    def prepare_model(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name, cache_dir=self.args.cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(self.args.model_name, cache_dir=self.args.cache_dir).to(self.device)
    
    def translate(self, text, src_lang, tgt_lang):
        messages = [{"role": "user", "content": f"Translate to {tgt_lang}: " + text}]
        
        encoded_input = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        
        output_tokens = self.model.generate(
            encoded_input.to(self.device),
            max_new_tokens=256,
            do_sample=False
        )
        translated_text = self.tokenizer.decode(
            output_tokens[0][len(encoded_input[0]):], skip_special_tokens=True
        )
        return translated_text
    
    def translate_constraint_soft(self, text, src_lang, tgt_lang, terms_dict, soft_penalty=0.8):
        messages = [{"role": "user", "content": f"Translate to {tgt_lang}: " + text}]
        
        encoded_input = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        
        tokens = []
        for translation in terms_dict:
            token = self.tokenizer(translation)['input_ids']
            tokens.append(token)
        
        logits_processor = TerminologyAwareLogitsProcessor(
            tokenizer=self.tokenizer,
            en_text=text,
            lang=tgt_lang,
            soft_penalty=soft_penalty,
            lang_dict=self.args.lang_dict,
            tokens = tokens,
        )
        
        output_tokens = self.model.generate(
            encoded_input.to(self.device),
            max_new_tokens=256,
            logits_processor = [logits_processor],
            do_sample=False
        )
        translated_text = self.tokenizer.decode(
            output_tokens[0][len(encoded_input[0]):], skip_special_tokens=True
        )
        return translated_text

            
class GoogleTranslator(Translator):
    def prepare_model(self):
        from googletrans import Translator as TLR
        self.model = TLR()
        
    def translate(self, text, src_lang, tgt_lang):
        max_tries = 5
        curr_tries = 0
        while True:
            try:
                translated_text = self.model.translate(
                    text,
                    dest=self.args.lang_dict[tgt_lang],
                    src=self.args.lang_dict[src_lang]
                )
                return translated_text.text
            except Exception as e:
                print(e)
                if curr_tries < max_tries:
                    curr_tries += 1
                    time.sleep(5)
                else:
                    return None
