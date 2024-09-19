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

client = AzureOpenAI(
    azure_endpoint="https://yuncong.openai.azure.com/",
    api_key=os.getenv('AZURE_OPENAI_KEY'),
    api_version="2024-02-15-preview",
)

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
def call_openai_api(sys_prompt, contents) -> Optional[str]:
    max_tries = 5
    retry_count = 0
    formated_content = format_content(contents)
    message_text = [
        {"role": "system", "content": sys_prompt},
        {"role": "user","content": formated_content}
    ]
    while retry_count < max_tries:
        try:
            completion = client.chat.completions.create(
                model="gpt-4o",  # model = "deployment_name"
                messages=message_text,
                temperature=0.7,
                max_tokens=4096,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
            )
            return completion.choices[0].message.content
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

# encode tensor images to base64 format
def encode_tensor2base64(img):
    img = Image.fromarray(img)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    return img_base64


def format_confidence_question(question, img):
    sys_prompt = "Task: You are an agent in an indoor scene tasked with answering questions by observing the surroundings and exploring the environment."
    content = []
    
    text = "Here is the current view of the scene."
    img = encode_tensor2base64(img)
    content.append((text, img))
    text = f"\nConsider the question: '{question}'. Are you confident about answering the question with the current view? Answer with score 0-10, where 0 is not confident at all and 10 is very confident.\n"
    content.append((text,))
    text = "you should return a score between 0 and 10.\n"
    text += "you can show the reason for your confidence score but put it in a new line after the choice.\n"
    return sys_prompt, content

def get_confidence(question, img):
    sys_prompt, content = format_confidence_question(question, img)
    retry_limit = 3
    while retry_limit > 0:
        response = call_openai_api(sys_prompt, content)
        # parse the response
        if response is not None:
            response = response.strip()
            if "\n" in response:
                response = response.split("\n")
                response, reason = response[0], response[-1]
        else:
            reason = ""
        response = response.lower()
        try:
            confidence = float(response)
            if confidence >= 0 and confidence <= 10:
                logging.log(f"Reason for confidence{confidence/10}: {reason}")
                return confidence/10
            else:
                logging.info("Invalid response, retrying")
        except ValueError:
            logging.info("Invalid response, retrying")
        retry_limit -= 1
    
    # no valid reponse, not sure about the question
    return 0

def format_choose_direction(question, img, candidates):
    sys_prompt = "Task: You are an agent in an indoor scene tasked with answering questions by observing the surroundings and exploring the environment."
    content = []
    
    text = "Here is the current view of the scene."
    img = encode_tensor2base64(img)
    content.append((text, img))
    text = f"\nConsider the question: '{question}', and you will explore the environment for answering it.\nWhich direction (black letters on the image) would you explore then?\n"
    text += f"Answer with following directions: {','.join(candidates)}\n and scores from 0-10, where 0 is not interested in exploring this direction and 10 is very interested in exploring\n"
    text += f"Each (direction,score) pair should be printed in a new line. \n"
    content.append((text,))
    
    text = "For example, if you are given candidates choices: A, B, C\n"
    text = "You can answer like this: A 5\n B 7\n C 3\n \n"
    text = "This means you think B is the most interesting direction to explore, while A and C may also be potential answers.\n"
    content.append((text,))
    return sys_prompt, content

def get_directions(question, img, candidates):
    sys_prompt, content = format_choose_direction(question, img, candidates)
    retry_limit = 3
    while retry_limit > 0:
        response = call_openai_api(sys_prompt, content)
        if response is None:
            logging.info("Invalid response, retrying")
            retry_limit -= 1
            continue
        # parse the response
        response = response.split("\n")
        directions = {}
        for r in response:
            r = r.split()
            if len(r) == 2:
                directions[r[0]] = float(r[1])
        if list(directions.keys()) != candidates:
            logging.info("Invalid response, retrying")
            retry_limit -= 1
            continue
        scores = np.array(list(directions.values()))
        scores = np.exp(scores) / np.sum(np.exp(scores))
        return scores
    
    return np.ones(actual_num_prompt_points) / actual_num_prompt_points

def format_global_selection(question, img):
    sys_prompt = "Task: You are an agent in an indoor scene tasked with answering questions by observing the surroundings and exploring the environment."
    content = []
    
    text = "Here is the current view of the scene."
    img = encode_tensor2base64(img)
    content.append((text, img))
    
    text = f"\nConsider the question: '{question}', and you will explore the environment for answering it. Is there any direction shown in the image worth exploring?\n"
    text = "Answer with score 0-10, where 0 is no directions worth exploring and 10 is very confident that there is direction worth exploring\n"
    text += "you can show the reason for your confidence score but put it in a new line after the choice.\n"
    
    return sys_prompt, content

def get_global_value(question, img):
    sys_prompt, content = format_global_selection(question, img)
    retry_limit = 3
    while retry_limit > 0:
        response = call_openai_api(sys_prompt, content)
        # parse the response
        if response is not None:
            response = response.strip()
            if "\n" in response:
                response = response.split("\n")
                response, reason = response[0], response[-1]
        else:
            reason = ""
        response = response.lower()
        try:
            confidence = float(response)
            if confidence >= 0 and confidence <= 10:
                logging.log(f"Reason for confidence{confidence/10}: {reason}")
                return confidence/10
            else:
                logging.info("Invalid response, retrying")
        except ValueError:
            logging.info("Invalid response, retrying")
        retry_limit -= 1
    
    return 0
    
            
    