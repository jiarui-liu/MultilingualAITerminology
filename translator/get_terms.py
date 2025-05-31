import json
import os
import re
import pandas as pd
from Levenshtein import distance
from fuzzywuzzy import fuzz
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

def normalized_levenshtein_similarity(str1, str2):
    # Calculate Levenshtein distance
    dist = distance(str1, str2)
    # Normalize by the length of the longer string
    max_len = max(len(str1), len(str2))
    return 1 - dist / max_len if max_len else 1.0

def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text.lower())
    token_indices = []
    current_position = 0
    for token in tokens:
        start_idx = text.lower().find(token, current_position)
        end_idx = start_idx + len(token)
        token_indices.append((token, start_idx, end_idx))
        current_position = end_idx  # Update current position for the next search

    lemmatized_tokens = [lemmatizer.lemmatize(token) for token, _, _ in token_indices]
    return lemmatized_tokens, token_indices

def get_substrings(words, n):
    substrings = []
    for i in range(len(words) - n + 1):
        substring = " ".join(words[i:i+n])
        substrings.append(substring)
    return substrings

def get_original_indices(lemmatized_tokens, token_indices, substring):
    substring = substring.split()
    for i in range(len(token_indices) - len(substring) + 1):
        # Check if the n adjacent lemmatized tokens match
        match = True
        for j in range(len(substring)):
            if lemmatized_tokens[i + j] != substring[j]:
                match = False
                break
        if match:
            # Get the original start and end indices of the matched sequence
            start_idx = token_indices[i][1]
            end_idx = token_indices[i + len(substring) - 1][2]
            return start_idx, end_idx
    return None

# too loose, adandoned
def find_terminology(in_paragraph, terminology):
    paragraph = in_paragraph
    paragraph, indices = preprocess_text(paragraph)
    terminology_lst, _ = preprocess_text(terminology)
    terminology = " ".join(terminology_lst)
    token_counts_paragraph = len(paragraph)
    token_counts = len(terminology_lst)
    substrings = get_substrings(paragraph, token_counts)
    if token_counts < token_counts_paragraph:
        substrings += get_substrings(paragraph, token_counts + 1)
    if token_counts > 0:
        substrings += get_substrings(paragraph, token_counts - 1)
    results = []
    for substring in substrings:
        if normalized_levenshtein_similarity(terminology, substring) > 0.9:
            sub_indices = get_original_indices(paragraph, indices, substring)
            print(sub_indices, terminology, substring)
            if sub_indices is not None:
                results.append(in_paragraph[sub_indices[0]:sub_indices[1]])
    return results


class TermCollector:
    def __init__(self, term_file_path, tgt_langs):
        self.terms_dict = {}
        for tgt_lang in tgt_langs:
            terms = pd.read_csv(os.path.join(term_file_path, tgt_lang + ".csv")).to_dict(orient='records')
            for term in terms:
                if term['English'] in self.terms_dict:
                    self.terms_dict[term['English'].lower()][tgt_lang] = term[tgt_lang]
                else:
                    self.terms_dict[term['English'].lower()] = {
                        tgt_lang: term[tgt_lang]
                    }
        
        # sort terms by the number of words in the English term in descending order
        self.sorted_terms = sorted(self.terms_dict.keys(), key = lambda term: len(term.split()), reverse=True)
    
    def find_terminology(self, in_paragraph):
        # Create a list to hold matched terms
        matched_terms = []
        matched_indices = set()  # To track matched word positions

        words = word_tokenize(in_paragraph.lower())
        for term in self.sorted_terms:
            term_words = word_tokenize(term)
            term_length = len(term_words)
            # Iterate over the words in the paragraph
            for i in range(len(words) - term_length + 1):
                # Check if the segment of words matches the term
                if words[i:i + term_length] == term_words:
                    # Ensure none of the words have been matched before to avoid overlaps
                    if all(idx not in matched_indices for idx in range(i, i + term_length)):
                        matched_terms.append(term)
                        matched_indices.update(range(i, i + term_length))  # Mark indices as matched

        return set(matched_terms)
    
    def find_terminology_with_indices(self, in_paragraph):
        # Create a list to hold matched terms
        matched_terms_with_indices = []
        matched_indices = set()  # To track matched word positions
        
        words = word_tokenize(in_paragraph.lower())
        for term in self.sorted_terms:
            term_words = word_tokenize(term)
            term_length = len(term_words)
            # Iterate over the words in the paragraph
            for i in range(len(words) - term_length + 1):
                # Check if the segment of words matches the term
                if words[i:i + term_length] == term_words:
                    # Ensure none of the words have been matched before to avoid overlaps
                    if all(idx not in matched_indices for idx in range(i, i + term_length)):
                        matched_terms_with_indices.append([term, [i, i+term_length]])
                        matched_indices.update(range(i, i + term_length))  # Mark indices as matched

        return matched_terms_with_indices
