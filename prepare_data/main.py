import json
import concurrent.futures
from threading import Lock
import os
import prepare_data.config as config
from prepare_data.req import LLM


def make_label_one(query, response1, response2, point, llm):
    """
    Call LLM once to determine which is better in terms of {point} \
        for {response1} and {response2} under the {query}
    """
    
    prompt = f"给定提问和两个回复，判断回复1和回复2在【{point}】方面哪个更好。\n【提问】：{query}\n\
        【回复1】：{response1}\n【回复2】：{response2}"
    res = llm.gen_res(prompt)
    position1 = res.find('1')  # If '1' appears first, then {response1} is considered better
    position2 = res.find('2')
    if position1 == -1 and position2 == -1:
        return None
    if position1 == -1:
        return 2
    if position2 == -1:
        return 1
    if position1 < position2:
        return 1
    else:
        return 2
    

def make_label(query, response1, response2, point, llm):
    """
    Swap the positions of responses and call twice, retaining only two results is consistent
    """

    res1 = make_label_one(query, response1, response2, point, llm)
    res2 = make_label_one(query, response2, response1, point, llm)
    if res1 is None or res2 is None:
        return None
    if res1 == 1 and res2 == 2:
        return [1, 2]
    if res1 == 2 and res2 == 1:
        return [2, 1]
    return None


def worker(json_line, lock, llm):
    """
    For a sample, obtain the preference labels for each capability point corresponding to its category
    """

    json_data = json.loads(json_line)
    cat_Chinese = json_data['label']
    cat = None
    for key, item in config.categories.items():
        if cat_Chinese == item['Chinese']:
            cat = key
            break
    if cat is None:
        return

    print(f'begin a sample of {cat}')
    points_Chinese = config.categories[cat]['points_Chinese']
    points = config.categories[cat]['points']
    points_data = []
    query = json_data['src'][-1]
    for i, response1 in enumerate(json_data['response']):
        for j, response2 in enumerate(json_data['response']):
            if i >= j:
                continue
            for index, point in enumerate(points):
                label = make_label(query, response1, response2, points_Chinese[index], llm)
                if label is not None:
                    points_data.append({'query': query,
                                        'responses': [response1, response2],
                                        'rank': label,
                                        'point': point})
    with lock:
        with open(os.path.join(config.phasedata_dir, cat, 'points.jsonl'), 'a') as f:
            for data in points_data:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
        with open(os.path.join(config.phasedata_dir, cat, 'cat.jsonl'), 'a') as f:
            f.write(json_line)
    print(f'end a sample of {cat}')


def prepare_data():
    """
    Concurrent requests to obtain all capability point preference labels
    """

    for cat in config.categories.keys():
        if os.path.exists(os.path.join(config.phasedata_dir, cat)):
            print('already having capability point preference labels, passed!')
            return
    for cat in config.categories.keys():
        os.makedirs(os.path.join(config.phasedata_dir, cat), exist_ok=True)

    llm = LLM()
    with open(os.path.join(config.phasedata_dir, 'routed.jsonl'), 'r') as f:
        json_lines = f.readlines()
    lock = Lock()
    with concurrent.futures.ThreadPoolExecutor(max_workers=config.num_workers) as executor:
        futures = [executor.submit(worker, json_line, lock, llm) for json_line in json_lines]
        concurrent.futures.wait(futures)
