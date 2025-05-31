prompt_refine = """You are tasked with improving the quality of translations containing AI-related terminologies. Given an English text and its machine-translated output in {tgt_lang}, your role is to analyze the translation for grammatical and morphological errors, such as incorrect word forms, cases, or agreement issues. You cannot change the content or meaning of the text; your task is strictly limited to fixing grammatical issues and ensuring proper linguistic structure.

Term dictionary:
{term_str}

{src_lang} text:
{text}

{tgt_lang} translation:
{result}

Provided the updated translation only.
"""