from base_config import *

num_workers = 1  # if use local LLM, set it to 1

categories = {
    'roleplay': {
        'Chinese': '角色扮演',
        'points_Chinese': ['人设和情感带入', '对话感', '共情能力', '关系特点体现', '个性化特征体现', '内容丰富性'],
        'points': ['character_setting_and_emotional_investment', 'conversational_sense', 'empathy_ability', \
            'manifestation_of_relationship_traits', 'personalized_characteristic_expression', 'content_richness']
    },
    'chat': {
        'Chinese': '闲聊',
        'points_Chinese': ['对话感', '主动性', '情感表达', '共情能力', '内容丰富性'],
        'points': ['conversational_sense', 'proactivity', 'emotion_expression', 'empathy_ability', 'content_richness']
    },
    'subj_qa': {
        'Chinese': '主观知识问答',
        'points_Chinese': ['说服力', '逻辑性', '观点丰富度', '知识面广度', '问题针对性'],
        'points': ['convincing_ability', 'logic', 'viewpoint_richness', 'breadth_of_knowledge', 'question_specific']
    },
    'obj_qa': {
        'Chinese': '客观知识问答',
        'points_Chinese': ['正确性', '客观程度', '推理能力', '逻辑性', '知识面深度', '问题针对性'],
        'points': ['correctness', 'objectiveness', 'reasoning_ability', 'logic', 'depth_of_knowledge', 'question_specific']
    },
    'text': {
        'Chinese': '文本创作',
        'points_Chinese': ['意图符合程度', '表达能力', '可读性', '内容丰富性', '逻辑性'],
        'points': ['intent_conformity', 'expressiveness', 'readability', 'content_richness', 'logic']
    }
}
