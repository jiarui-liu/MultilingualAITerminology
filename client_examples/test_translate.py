import requests

url = "http://localhost:8765/translate"
payload = {
    "text": "1: For all the methods, we used <mark>10-fold cross validation</mark> (i.e., each fold we have 556 training and 62 test samples) to tune free parameters, e.g., the kernel form and parameters for GPOR and LapSVM. Note that all the alternative methods stack X and Z together into a whole data matrix and ignore their heterogeneous nature.<br>2: Features associated one-to-one with a vertical (Clarity, ReDDE, the query likelihood given the vertical's query-log and Soft.ReDDE) were normalized across verticals before scaling. Supervised training/testing was done via <mark>10-fold cross validation</mark>. Parameter τ was tuned for each training fold on the same 500 query validation set used for our single feature baselines.<br>",
    "src_lang": "English",
    "tgt_lang": "Chinese",
    "mode": "direct",
    "seamless": ""
}
headers = {
    "Content-Type": "application/json; charsetUTF-8"
}

try:
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()  # Raises HTTPError if the request returned an unsuccessful status code
    data = response.json()
    print("Translated Text:", data["translated_text"])
except requests.exceptions.RequestException as e:
    print("Error:", e)
