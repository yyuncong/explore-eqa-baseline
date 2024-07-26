import json
from collections import defaultdict

def scene2questionid(path):
    with open(path,'r') as f:
        questions = json.load(f)
    scene_questions = defaultdict(list)
    for q in questions:
        scene_questions[q['episode_history'].split('-')[1]].append(q['question_id'])
    print(scene_questions)
    

def gather_scene_info(path):
    with open(path,'r') as f:
        questions = json.load(f)
    scenes = {}
    for q in questions:
        scenes[q['episode_history'].split('-')[1]] = q['episode_history']
    for k,v in scenes.items():
        print(k,v)
    return scenes
    
if __name__ == "__main__":
    #scene2questionid("data/generated_questions-eval.json")
     
    gather_scene_info("data/filtered_data.json")