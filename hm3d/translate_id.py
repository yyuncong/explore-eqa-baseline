import json
from collections import defaultdict

def scene2questionid(path):
    with open(path,'r') as f:
        questions = json.load(f)
    scene_questions = defaultdict(list)
    for q in questions:
        scene_questions[q['episode_history'].split('-')[1]].append(q['question_id'])
    for k,v in scene_questions.items():
        for vv in v:
            print(k,vv)
    #print(scene_questions)
    

def gather_scene_info(path):
    with open(path,'r') as f:
        questions = json.load(f)
    scenes = {}
    for q in questions:
        scenes[q['episode_history'].split('-')[1]] = q['episode_history']
    for k,v in scenes.items():
        print(k,v)
    return scenes

scene_list = [
    '00871-VBzV5z6i1WS',
    '00808-y9hTuugGdiq',
    '00821-eF36g7L6Z9M',
    '00847-bCPU9suPUw9',
    '00844-q5QZSEeHe5g',
    '00823-7MXmsvcQjpJ',
    '00862-LT9Jq6dN3Ea',
    '00861-GLAQ4DNUx5U',
    '00823-7MXmsvcQjpJ',
    '00827-BAbdmeyTvMZ']
#    '00802-wcojb4TFT35'
#]

def select_test_scene(question_path, scene_list):
    with open(question_path,'r') as f:
        questions = json.load(f)
    test_questions = []
    for q in questions:
        if q['episode_history'] in scene_list:
            test_questions.append(q)
    with open('data/consern_data.json','w') as f:
        json.dump(test_questions,f)
    return test_questions

if __name__ == "__main__":
    #scene2questionid("data/filtered_data.json")
    #gather_scene_info("data/filtered_data.json")
    select_test_scene("data/filtered_data.json",scene_list)
    '''
    with open('data/consern_data.json','r') as f:
        questions = json.load(f)
    print(len(questions))
    '''