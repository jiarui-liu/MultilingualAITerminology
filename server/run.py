from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import difflib
from nltk.tokenize import sent_tokenize
import nltk
import argparse
import uvicorn
from server.model import Translator

# Define the request body structure
class TranslationRequest(BaseModel):
    text: str
    src_lang: str
    tgt_lang: str 
    mode: str
    seamless : str = ""

# Define the request body structure
class MarkRequest(BaseModel):
    seamless : str
    prompt : str
    lang : str

# Define the response structure
class MarkResponse(BaseModel):
    marked_translations: list[str]
    
# Define the response structure
class TranslationResponse(BaseModel):
    translated_text: str


# Start the server
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_mt", type=str, choices=['seamless', 'gpt-4o-mini', 'googletrans'], default="seamless")
    parser.add_argument("--model_refine", type=str, choices=['gpt-4o-mini'], default="gpt-4o-mini")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--openai_key_path", type=str, default=None)
    parser.add_argument("--term_path", type=str, default="./data/")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    
    
    # Define the FastAPI app
    app = FastAPI(debug=True)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Adjust to specific domains if needed
        allow_methods=["*"],  # Allow all HTTP methods
        allow_headers=["*"],  # Allow all headers
    )
    
    translator = Translator(args)

    # API endpoint for translation
    @app.post("/translate", response_model=TranslationResponse)
    async def translate_text(request: TranslationRequest):
        try:
            # Perform the translation
            translated_text = translator.translate(request.text, request.src_lang, request.tgt_lang, request.mode, request.seamless)
            return TranslationResponse(translated_text=translated_text)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        
    # API endpoint for marking translations
    @app.post("/mark", response_model=MarkResponse)
    async def mark_text(request: MarkRequest):
        try:
            d = difflib.Differ()
            # for chinese and japanese compare character by character
            if (request.lang == "zh" or request.lang == "ja"):
                seamlessWords = list(request.seamless.replace("  ", " "))
                promptWords = list(request.prompt.replace("。", ".").replace("、", ",").replace("，", ",").replace("  ", " "))

                space = ""
            # for other languages compare word by word
            else:
                seamlessWords = request.seamless.replace(" ،", "،").replace("  ", " ").split()
                promptWords = request.prompt.replace(" ،", "،").replace("  ", " ").split()
                space = " "
                
            diff = d.compare(seamlessWords, promptWords)
            
            wordLst = []
            for d in diff:
                wordLst.append(d)
                
            # create final marked texts
            markedSeamless = ""
            markedPrompt = ""
            
            i = 0
            while (i < len(wordLst)):
                word = wordLst[i]

                # if question mark ignore
                if word[0] == "?":
                    i = i + 1
                    
                # if space automatically add
                elif word[0] == " ":
                    markedSeamless += space + word[2:]
                    markedPrompt += space + word[2:]
                    i = i + 1
                    
                # if -, mark and add to seamless translation
                elif word[0] == "-":
                    # fix to prevent highlighting of non alphanumeric characters
                    if not word[2:].isalnum():
                        markedSeamless += space + word[2:]
                        i=i+1
                    else:
                        j = i
                        marked = "<mark style='background-color: #FFCCCB'>"
                        while (j < len(wordLst) and wordLst[j][0] == "-"):
                            marked += space + wordLst[j][2:]
                            j += 1
                        
                        i = j
                        markedSeamless += space + marked + "</mark>"

                # if +, mark and add to prompt translation
                elif word[0] == "+":
                    # fix to prevent highlighting of non alphanumeric characters
                    if not word[2:].isalnum():
                        markedPrompt += space + word[2:]
                        i = i+1
                    else:
                        j = i
                        marked = "<mark style='background-color: #90EE90'>"
                        while (j < len(wordLst) and wordLst[j][0] == "+"):
                            marked += space + wordLst[j][2:]
                            j += 1
                        i = j
                        markedPrompt += space + marked + "</mark>"

            return MarkResponse(marked_translations=[markedSeamless.replace("  ", " "), markedPrompt.replace("  ", " ")])
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    uvicorn.run(app, host="0.0.0.0", port=args.port)