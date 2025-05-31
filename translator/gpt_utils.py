import time
import openai
import json

def gpt_completion(
    prompt,
    openai_setup_func,
    model="gpt-4o-mini",
    temperature=0,
    return_response=False,
    max_tokens=1024,
    verbose=False,
    out_f=None,
):
    openai_setup_func()
    
    client = openai.chat.completions
    max_retries = 5
    curr_retries = 0
    
    messages = [{
        'role': 'user',
        'content': prompt,
    }]
    
    if verbose:
        print("###### Prompt starts..")
        print(prompt)
        print("###### Prompt ends..")
        
    while True:
        try:
            response = client.create(
                model = model,
                messages = messages,
                max_tokens = max_tokens,
                temperature = temperature,
            )
            if return_response:
                return response
            response = response.choices[0].message.content

            if verbose:
                print("###### Response starts..")
                print(response)
                print("###### Response ends..")
            
            if out_f is not None:
                json.dump({"model": model, "input": prompt, "response": response}, out_f)
                out_f.write('\n')
                out_f.flush()
            
            return response
        
        except Exception as e:
            print(f"ERROR: Can't invoke '{model}'. Reason: {e}")
            if curr_retries >= max_retries:
                print("EXITING...")
                exit(1)
            else:
                print("RETRYING...")
                curr_retries += 1
                time.sleep(5)