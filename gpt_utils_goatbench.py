import openai
from openai import AzureOpenAI
from PIL import Image
import base64
from io import BytesIO
import os
import time
from typing import Optional
import logging
import numpy as np
import tiktoken

client = AzureOpenAI(
    azure_endpoint="https://jiaben.openai.azure.com/",
    api_key=os.getenv('AZURE_OPENAI_KEY'),
    api_version="2024-02-15-preview",
)
tokenizer = tiktoken.get_encoding("o200k_base")

def format_content(contents):
    formated_content = []
    for c in contents:
        formated_content.append({"type": "text", "text": c[0]})
        if len(c) == 2:
            formated_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{c[1]}",  # yh: previously I always used jpeg format. The internet says that jpeg is smaller in size? I'm not sure.
                        "detail": "high"
                     }
                }
            )
    return formated_content
    
# send information to openai
def call_openai_api(sys_prompt, contents, choices):
    max_tries = 5
    retry_count = 0
    formated_content = format_content(contents)
    message_text = [
        {"role": "system", "content": sys_prompt},
        {"role": "user","content": formated_content}
    ]

    choices_ids = [tokenizer.encode(c)[0] for c in choices]
    logit_bias = {str(i): 100 for i in choices_ids}

    while retry_count < max_tries:
        try:
            completion = client.chat.completions.create(
                model="gpt-4o",  # model = "deployment_name"
                messages=message_text,
                temperature=0,
                max_tokens=1,
                logprobs = True,
                top_logprobs = 5,
                logit_bias=logit_bias,
            )
            return completion
        except openai.RateLimitError as e:
            print("Rate limit error, waiting for 60s")
            time.sleep(30)
            retry_count += 1
            continue
        except Exception as e:
            print("Error: ", e)
            time.sleep(60)
            retry_count += 1
            continue

    return None

def parse_probs(response):
    top_logprobs = response.choices[0].logprobs.content[0].top_logprobs
    #print(top_logprobs)
    prob_dict = {
        log_probs.token: log_probs.logprob
        for log_probs in top_logprobs
    }
    return prob_dict
        
# encode tensor images to base64 format
def encode_tensor2base64(img):
    #img = Image.fromarray(img)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    return img_base64

def format_confidence_question(question, ref_image, obs_image):
    sys_prompt = "Task: You are an agent in an indoor scene tasked with finding a target object by observing the surroundings and exploring the environment."
    content = []
    
    text = "Here is the current view of the scene."
    obs_image = encode_tensor2base64(obs_image)
    content.append((text, obs_image))
    if ref_image is None:
        text = f"\nConsider the question: '{question}'."
        content.append((text,))
    else:
        text = f"\nConsider the question: '{question}'."
        ref_image = encode_tensor2base64(ref_image)
        content.append((text, ref_image))
    text = "\nAre you confident that you have arrived at the target object described in the question, and the target object is in the current view? Answer with Yes/No \n"
    content.append((text,))
    #text = "you should return a score between 0 and 10.\n"
    #text += "you can show the reason for your confidence score but put it in a new line after the choice.\n"
    return sys_prompt, content

def get_confidence(question, ref_image, obs_image):
    sys_prompt, content = format_confidence_question(question, ref_image, obs_image)

    logging.info(f"\n##########\nGet confidence prompt:")
    message = ''
    for c in content:
        message += c[0]
        if len(c) == 2:
            message += f"[{c[1][:10]}...]"
    logging.info(message)

    retry_limit = 3
    while retry_limit > 0:
        response = call_openai_api(sys_prompt, content, ["Yes", "No"])
        if response is None:
            logging.info("Invalid response, retrying")
            retry_limit -= 1
            continue
        # parse the response
        log_probs = parse_probs(response)
        if "Yes" not in log_probs and "No" not in log_probs:
            logging.info("Invalid response, retrying")
            retry_limit -= 1
            continue
        if "Yes" not in log_probs:
            log_probs["Yes"] = -1e4
        if "No" not in log_probs:
            log_probs["No"] = -1e4
        probs = np.array([log_probs["Yes"], log_probs["No"]])
        probs = np.exp(probs) / np.sum(np.exp(probs))
        return probs
    
    # no valid reponse, not sure about the question
    return 0

def format_choose_direction(question, ref_image, obs_image, candidates):
    sys_prompt = "Task: You are an agent in an indoor scene tasked with finding a target object by observing the surroundings and exploring the environment."
    content = []
    
    text = "Here is the current view of the scene."
    obs_image = encode_tensor2base64(obs_image)
    content.append((text, obs_image))
    text = f"\nConsider the question: '{question}'."
    if ref_image is not None:
        ref_image = encode_tensor2base64(ref_image)
        content.append((text, ref_image))
    else:
        content.append((text,))
    text = "\nYou will explore the environment for finding the target object described in the question. So which direction (black letters marked on the current view image) would you explore then?\n"
    text += f"Answer with following directions: {','.join(candidates)}\n"
    text += f"Please return with letter "
    for c in candidates[:-1]:
        text += f"{c}, "
    text += f"or {candidates[-1]} only.\n"
    content.append((text,))
    
    '''
    text = "For example, if you are given candidates choices: A, B, C\n"
    text = "You can answer like this: A 5\n B 7\n C 3\n \n"
    text = "This means you think B is the most interesting direction to explore, while A and C may also be potential answers.\n"
    content.append((text,))
    '''
    return sys_prompt, content

def get_directions(question, ref_image, obs_image, candidates):
    sys_prompt, content = format_choose_direction(question, ref_image, obs_image, candidates)

    logging.info(f"\n##########\nGet direction prompt:")
    message = ''
    for c in content:
        message += c[0]
        if len(c) == 2:
            message += f"[{c[1][:10]}...]"
    logging.info(message)

    retry_limit = 3
    while retry_limit > 0:
        response = call_openai_api(sys_prompt, content, candidates)
        if response is None:
            logging.info("Invalid response, retrying")
            retry_limit -= 1
            continue
        # parse the response
        log_probs = parse_probs(response)
        if set(candidates).issubset(set(log_probs.keys())):
            scores = np.zeros(len(candidates))
            for i, c in enumerate(candidates):
                if c in log_probs:
                    scores[i] = log_probs[c]
                else:
                    scores[i] = -1e4
            scores = np.exp(scores) / np.sum(np.exp(scores))
            logging.info(f"VLM output: {log_probs}")
            logging.info(f"VLM scores: {scores}")
        else:
            logging.info("Not all choices considered, retrying")
            retry_limit -= 1
            continue
        return scores
    
    return np.ones(len(candidates)) / len(candidates)

def format_global_selection(question, ref_image, obs_image):
    sys_prompt = "Task: You are an agent in an indoor scene tasked with finding a target object by observing the surroundings and exploring the environment."
    content = []
    
    text = "Here is the current view of the scene."
    obs_image = encode_tensor2base64(obs_image)
    content.append((text, obs_image))
    
    text = f"\nConsider the question: '{question}'."
    if ref_image is not None:
        ref_image = encode_tensor2base64(ref_image)
        content.append((text, ref_image))
    else:
        content.append((text,))

    text = "\nYou will explore the environment for finding the target object described in the question. Is there any direction shown in the image worth exploring?\n"
    text += "Answer with \"Yes\" or \"No\"\n"
    content.append((text,))
    return sys_prompt, content

def get_global_value(question, ref_image, obs_image):
    sys_prompt, content = format_global_selection(question, ref_image, obs_image)

    logging.info(f"\n##########\nGet global value prompt:")
    message = ''
    for c in content:
        message += c[0]
        if len(c) == 2:
            message += f"[{c[1][:10]}...]"
    logging.info(message)

    retry_limit = 3
    while retry_limit > 0:
        response = call_openai_api(sys_prompt, content, ["Yes", "No"])
        if response is None:
            logging.info("Invalid response, retrying")
            retry_limit -= 1
            continue
        # parse the response
        #log_probs = response['choices'][0]['logprobs']['top_logprobs'][0]
        log_probs = parse_probs(response)
        if "Yes" not in log_probs and "No" not in log_probs:
            logging.info("Invalid response, retrying")
            retry_limit -= 1
            continue
        probs = np.array([log_probs["Yes"], log_probs["No"]])
        probs = np.exp(probs) / np.sum(np.exp(probs))
        return probs
    
    return 0
    
            
    