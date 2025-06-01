# Translation Server (used alongside [ACL Antho Mod](https://github.com/ImanOu123/acl-anthology-mod))

From the MultilingualAITerminology directory, run 
```
python3 -m server.run
```

You need an OpenAPI API key in your environment (for the prompt refinement 
translation feature) as follows:
```
openai_api_key='KEY_HERE'
```

In addition, you need the requirements in [/server/requirements.txt](https://github.com/jiarui-liu/MultilingualAITerminology/blob/main/server/requirements.txt) and to install 'wordnet' as follows:
```
import nltk
nltk.download('wordnet')
```

Finally, you need to download the dataset that the prompt refinment uses as follows:
```
python3 download_dataset.py
```

The server will run on default on https://localhost:8765.

There are two main POST requests that can be made on this server:

## Translate

This is to translate a text using Seamless (**the default direct translation method**)
or the prompt refinement method. <br/>

This POST request takes in the following arguments: 
+ text: the text to translate
+ src_lang: source language to translate from
+ tgt_lang: target language to translate to
+ mode: "direct" or "term_aware" <br/>
Setting the mode to "direct" will translate 
the text using Seamless and return the result. Setting the mode to "term_aware" 
will translate the text using the prompt refinement method using gpt-4o-mini.
+ seamless: the Seamless translation of the text. <br/>
The translator will translate using the Seamless translator regardless
if the mode is set to "direct" or "term_aware" (because term_aware uses the 
Seamless translation). This is used with the "term_aware" mode if you already have 
the Seamless translation available. 

An example of this request is:

```
const res = await fetch("http://127.0.0.1:8765/translate", {
    method: "POST",
    body: JSON.stringify({
        text: text,
        src_lang: "English",
        tgt_lang: "Arabic",
        mode: "direct",
        seamless : ""
    }),
    headers: {"Content-Type": "application/json; charsetUTF-8"}
});
```

## Mark

This is used alongside the website demonstration in order to highlight 
the differences between the Seamless and prompt refinement translations. The output
is HTML-formatted text. 

This POST request takes in the following arguments: 
+ seamless: the Seamless translation
+ prompt: the prompt refinement translation
+ lang: the language the text has been translated into

An example of this request is:

```
const res = await fetch("http://127.0.0.1:8765/mark", {
    method: "POST",
    body: JSON.stringify({
        seamless : trans1,
        prompt : trans2,
        lang : lang
    }),
    headers: {"Content-Type": "application/json; charsetUTF-8"}
    });
```