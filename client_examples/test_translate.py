import requests

url = "http://localhost:8765/translate"
payload = {
    "text": "Recent advances in few-shot question answering (QA) mostly rely on the power of pre-trained large language models (LLMs) and fine-tuning in specific settings. Although the pre-training stage has already equipped LLMs with powerful reasoning capabilities, LLMs still need to be fine-tuned to adapt to specific domains to achieve the best results. In this paper, we propose to select the most informative data for fine-tuning, thereby improving the efficiency of the fine-tuning process with comparative or even better accuracy on the open-domain QA task. We present MinPrompt, a minimal data augmentation framework for open-domain QA based on an approximate graph algorithm and unsupervised question generation. We transform the raw text into a graph structure to build connections between different factual sentences, then apply graph algorithms to identify the minimal set of sentences needed to cover the most information in the raw text. We then generate QA pairs based on the identified sentence subset and train the model on the selected sentences to obtain the final model. Empirical results on several benchmark datasets and theoretical analysis show that MinPrompt is able to achieve comparable or better results than baselines with a high degree of efficiency, bringing consistent improvements in F-1 scores.",
    "src_lang": "English",
    "tgt_lang": "French",
    "mode": "term_aware",
    "seamless": "Les progrès récents dans la réponse à quelques questions (QA) reposent principalement sur la puissance des grands modèles linguistiques (LLM) pré-entraînés et sur le réglage précis dans des contextes spécifiques. Bien que la phase de pré-entraînement ait déjà doté les LLM de puissantes capacités de raisonnement, les LLM doivent encore être ajustés pour s'adapter à des domaines spécifiques pour obtenir les meilleurs résultats. Dans ce document, nous proposons de sélectionner les données les plus informatives pour le réglage fin, améliorant ainsi l'efficacité du processus de réglage fin avec une précision comparative ou même meilleure sur la tâche d'assurance qualité de domaine ouvert. Nous présentons MinPrompt, un cadre d'augmentation de données minimale pour l'assurance qualité de domaine ouvert basé sur un algorithme de graphe approximatif et une génération de questions non supervisées. Nous transformons le texte brut en une structure graphique pour construire des connexions entre différentes phrases factuelles, puis appliquons des algorithmes graphiques pour identifier l'ensemble minimal de phrases nécessaires pour couvrir le plus d'informations dans le texte brut. Nous générons ensuite des paires QA basées sur le sous-ensemble de phrases identifiées et formons le modèle sur les phrases sélectionnées pour obtenir le modèle final. Les résultats empiriques de plusieurs ensembles de données de référence et l'analyse théorique montrent que MinPrompt est capable d'obtenir des résultats comparables ou meilleurs que les résultats de base avec un haut degré d'efficacité, entraînant des améliorations constantes des scores F-1."
}
headers = {
    "Content-Type": "application/json; charsetUTF-8"
}

try:
    response = requests.post(url, json=payload, headers=headers)
    print(response.text)
    response.raise_for_status()  # Raises HTTPError if the request returned an unsuccessful status code
    data = response.json()
    print("Translated Text:", data["translated_text"])
except requests.exceptions.RequestException as e:
    print("Error:", e)
