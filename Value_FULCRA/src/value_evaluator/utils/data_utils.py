import os
import json
import random
from tqdm import tqdm

value_item_list = [
    "Be creative", "Be curious", "Have freedom of thought",
    "Be choosing own goals", "Be independent", "Have freedom of action", "Have privacy",
    "Have an exciting life", "Have a varied life", "Be daring",
    "Have pleasure", "Enjoying life", "Be self-indulgent",
    "Be ambitious", "Be successful", "Be capable", "Be influential", "Be intellectual", "Have authority",
    "Have social power", "Have wealth", "Have a social recognition", "Preserving my public image", "Observing social norms",
    "Have a sense of belonging", "Have a good health", "Have no debts", "Be neat and tidy", "Have family security",
    "Have a safe country", "Have a stable society",
    "Be respecting traditions", "Be holding religious faith",
    "Be obedient", "Be self-disciplined", "Moderate",
    "Be polite", "Be honoring parents and elders", "Be humble", "Accepting my portion in life",
    "Be helpful", "Be honest", "Be forgiving", "True friendship", "Mature love",
    "Be responsible", "Have loyalty towards friends",
    "Have equality", "Social justice", "Have a world at peace",
    "Be protecting the environment", "Have harmony with nature", "Have a world of beauty",
    "Be broadminded"
] # length: 54

value_item_definition_dict = {
    "Be creative": "valuing uniqueness and using imagination to create unique ideas or product",
    "Be curious": "interested in everything, seeking new knowledge, experiences and learning new things",
    "Have freedom of thought": "form one's own opinions",
    "Be choosing own goals": "selecting and pursuing own purposes and objectives",
    "Be independent": "being self-reliant, self-sufficient, doing everything by oneself, without depending on others",
    "Have freedom of action": "prioritizing the ability to make one's own choices and decisions",
    "Have privacy": "the right to have a privacy sphere, have a personal space and boundaries",
    "Have an exciting life": "stimulating experiences and adventures",
    "Have a varied life": "filled with challenge, novelty, change and diverse experience",
    "Be daring": "seeking adventure, risk, willing to take risks or engage in adventurous activities",
    "Have pleasure": "seeking gratification of desires and enjoyment",
    "Enjoying life": "enjoying food, sex, leisure, etc.",
    "Be self-indulgent": "doing pleasant things, engaging in activities that bring personal satisfaction",
    "Be ambitious": "being hard-working, aspiring, a strong desire of success",
    "Be successful": "achieving one's goals and accomplishments",
    "Be capable": "being competent, effective and efficient in various tasks",
    "Be influential": "having an impact on people and events",
    "Be intellectual": "be knowledgeable, perceptive, think logically and critically",
    "Have authority": "exercising the right to lead or command others",
    "Have social power": "controlling or dominating over others in social settings",
    "Have wealth": "material possessions, financial resources",
    "Have a social recognition": "being respected, approved and acknowledged by others", "Have social recognition": "being respected, approved and acknowledged by others",
    "Preserving my public image": "protecting my 'Face'",
    "Observing social norms": "observing social norms to protect my 'Face'",
    "Have a sense of belonging": "feeling that others care about me",
    "Have a good health": "not being sick physically or mentally", "Have health": "not being sick physically or mentally", "Have a healthy life": "not being sick physically or mentally",
    "Have no debts": "avoidance of indebtedness",
    "Be neat and tidy": "Keeping oneself and surrounding things clean and organized",
    "Have family security": "protecting my family", "Have security": "protecting my family",
    "Have a safe country": "protection of my nation from external threats",
    "Have a stable society": "ensuring social order and harmony",
    "Be respecting traditions": "preserving and valuing time-honored customs",
    "Be holding religious faith": "being devout and committed to one's religion",
    "Be obedient": "being dutiful, meeting obligations",
    "Be self-disciplined": "self-restraint, resistance to temptation",
    "Moderate": "avoiding extremes of feeling & action", "Be moderate": "avoiding extremes of feeling & action",
    "Be polite": "demonstrating courtesy, good manners",
    "Be honoring parents and elders": "showing respect and deference",
    "Be humble": "modest, self-effacing",
    "Accepting my portion in life": "submitting to life's circumstances", "Be accepting my portion in life": "submitting to life's circumstances",
    "Be helpful": "working for the welfare of others",
    "Be honest": "being genuine, sincere",
    "Be forgiving": "willing to pardon others",
    "True friendship": "close, supportive friends",
    "Mature love": "deep emotional & spiritual intimacy",
    "Be responsible": "being dependable and reliable",
    "Have loyalty towards friends": "being faithful to my friends and group members",
    "Have equality": "supporting equal rights and opportunities for all individuals",
    "Social justice": "correcting injustice, care for the weak", "Have social justice": "correcting injustice, care for the weak",
    "Have a world at peace": "striving a world free of war and conflict",
    "Be protecting the environment": "Safeguarding nature and its resources",
    "Have harmony with nature": "fitting into nature",
    "Have a world of beauty": "appreciating the beauty of nature and the arts",
    "Be broadminded": "being tolerant of diverse ideas and beliefs",
}

value_type_19_list = [
    "Self-direction-thought", "Self-direction-action",
    "Stimulation", "Hedonism", "Achievement",
    "Power-dominance", "Power-resources", "Face",
    "Security-personal", "Security-societal", "Tradition",
    "Conformity-rules", "Conformity-interpersonal", "Humility",
    "Benevolence-caring", "Benevolence-dependability",
    "Universalism-concern", "Universalism-nature", "Universalism-tolerance"
]

value_type_10_list = [
    "Self-direction",
    "Stimulation",
    "Hedonism",
    "Achievement",
    "Power",
    "Security",
    "Tradition",
    "Conformity",
    "Benevolence",
    "Universalism"
]

value_type_19_definition_dict = {
    "Self-direction-thought": "this value means freedom to cultivate one's own ideas and abilities. It can be characterized by several specific values: Be creative (valuing uniqueness and using imagination to create unique ideas or product), Be curious (interested in everything, seeking new knowledge, experiences and learning new things) and Have freedom of thought (form one's own opinions).",
    "Self-direction-action": "this value means freedom to determine one's own actions. It can be characterized by several specific values: Be choosing own goals (selecting and pursuing own purposes and objectives), Be independent (being self-reliant, self-sufficient, doing everything by oneself, without depending on others), Have freedom of action (prioritizing the ability to make one's own choices and decisions) and Have privacy (the right to have a privacy sphere, have a personal space and boundaries).",
    "Stimulation": "this value means excitement, novelty, and challenge in life. It can be characterized by several specific values: Have an exciting life (stimulating experiences and adventures), Have a varied life (filled with challenge, novelty, change and diverse experience) and Be daring (seeking adventure, risk, willing to take risks or engage in adventurous activities).",
    "Hedonism": "this value means pleasure or sensuous gratification for oneself. It can be characterized by several specific values: Have pleasure (seeking gratification of desires and enjoyment), Enjoying life (enjoying food, sex, leisure, etc.) and Be self-indulgent (doing pleasant things, engaging in activities that bring personal satisfaction).",
    "Achievement": "this value means personal success through demonstrating competence according to social standards. It can be characterized by several specific values: Be ambitious (being hard-working, aspiring, a strong desire of success), Be successful (achieving one's goals and accomplishments), Be capable (being competent, effective and efficient in various tasks), Be influential (having an impact on people and events) and Be intellectual (be knowledgeable, perceptive, think logically and critically).",
    "Power-dominance": "this value means power through exercising control over people. It can be characterized by several specific values: Have authority (exercising the right to lead or command others) and Have social power (controlling or dominating over others in social settings)",
    "Power-resources": "this value means power through control of material and social resources. It can be characterized by several specific values: Have wealth (material possessions, financial resources)",
    "Face": "this value means security and power through maintaining one's public image and avoiding humiliation. It can be characterized by several specific values: Have a social recognition (being respected, approved and acknowledged by others), Preserving my public image (protecting my 'Face') and Observing social norms (observing social norms to protect my 'Face')",
    "Security-personal": "this value means safety in one's immediate environment. It can be characterized by several specific values: Have a sense of belonging (feeling that others care about me), Have a good health (not being sick physically or mentally), Have health (not being sick physically or mentally), Have a healthy life (not being sick physically or mentally), Have no debts (avoidance of indebtedness), Be neat and tidy (Keeping oneself and surrounding things clean and organized) and Have family security (protecting my family)",
    "Security-societal": "this value means safety and stability in the wider society. It can be characterized by several specific values: Have a safe country (protection of my nation from external threats), Have a stable society (ensuring social order and harmony)",
    "Tradition": "this value means respect, commitment, and acceptance of the customs and ideas that one's culture or religion provides. It can be characterized by several specific values: Be respecting traditions (preserving and valuing time-honored customs) and Be holding religious faith (being devout and committed to one's religion)",
    "Conformity-rules": "this value means compliance with rules, laws, and formal obligations. It can be characterized by several specific values: Be obedient (being dutiful, meeting obligations), Be self-disciplined (self-restraint, resistance to temptation) and Moderate (avoiding extremes of feeling & action)",
    "Conformity-interpersonal": "this value means avoidance of upsetting or harming other people. It can be characterized by several specific values: Be polite (demonstrating courtesy, good manners), Be honoring parents and elders (showing respect and deference)",
    "Humility": "this value means recognizing one's insignificance in the larger scheme of things. It can be characterized by several specific values: Be humble (modest, self-effacing), Accepting my portion in life (submitting to life's circumstances)",
    "Benevolence-caring": "this value means devotion to the welfare of ingroup members. It can be characterized by several specific values: Be helpful (working for the welfare of others), Be honest (being genuine, sincere), Be forgiving (willing to pardon others), True friendship (close, supportive friends) and Mature love (deep emotional & spiritual intimacy)",
    "Benevolence-dependability": "this value means being a reliable and trustworthy member of the ingroup. It can be characterized by several specific values: Be responsible (being dependable and reliable) and Have loyalty towards friends (being faithful to my friends and group members)",
    "Universalism-concern": "this value means commitment to equality, justice, and protection for all people. It can be characterized by several specific values: Have equality (supporting equal rights and opportunities for all individuals), Social justice (correcting injustice, care for the weak) and Have a world at peace (striving a world free of war and conflict)",
    "Universalism-nature": "this value means preservation of the natural environment. It can be characterized by several specific values: Be protecting the environment (Safeguarding nature and its resources), Have harmony with nature (fitting into nature), Have a world of beauty (appreciating the beauty of nature and the arts)",
    "Universalism-tolerance": "this value means acceptance and understanding of those who are different from oneself. It can be characterized by several specific values: Be broadminded (being tolerant of diverse ideas and beliefs)"
}

value_type_10_definition_dict = {
    "Self-direction": "this value means independent thought and action-choosing, creating, exploring",
    "Stimulation": "this value means excitement, novelty, and challenge in life",
    "Hedonism": "this value means pleasure and sensuous gratification for oneself",
    "Achievement": "this value means personal success through demonstrating competence according to social standards",
    "Power": "this value means social status and prestige, control or demdominance over people and resources",
    "Security": "this value means safety, harmony, and stability of society, of relationships, and of self",
    "Tradition": "this value means respect, commitment, and acceptance of the customs and ideas that traditional culture or religion provide",
    "Conformity": "this value means restraint of actions, inclinations, and impulses likely to upset or harm others and violate social expectations or norms",
    "Benevolence": "this value means preservaion and enhancement of the welfare of people with whom one is in frequent personal contact",
    "Universalism": "this value means understanding, appreciation, tolerance, and protection for the welfare of all people and for nature",
}

value_item_2_type_19 = {
    "be creative": "Self-direction-thought", "be curious": "Self-direction-thought", "have freedom of thought": "Self-direction-thought",
    "be choosing own goals": "Self-direction-action", "be independent": "Self-direction-action", "have freedom of action": "Self-direction-action", "have privacy": "Self-direction-action",
    "have an exciting life": "Stimulation", "have a varied life": "Stimulation", "be daring": "Stimulation",
    "have pleasure": "Hedonism", "enjoying life": "Hedonism", "be self-indulgent": "Hedonism", "self-indulgent": "Hedonism",
    "be ambitious": "Achievement", "be successful": "Achievement", "be capable": "Achievement", "be influential": "Achievement", "be intellectual": "Achievement", "self-respect": "Achievement",
    "have authority": "Power-dominance", "have social power": "Power-dominance",
    "have wealth": "Power-resources",
    "have a social recognition": "Face", "preserving my public image": "Face", "observing social norms": "Face", "have social recognition": "Face",
    "have a sense of belonging": "Security-personal", "have a good health": "Security-personal", "have no debts": "Security-personal", "be neat and tidy": "Security-personal", "have family security": "Security-personal", "family security": "Security-personal",
    "have a safe country": "Security-societal", "have a stable society": "Security-societal", "have a stable social": "Security-societal",
    "be respecting traditions": "Tradition", "be holding religious faith": "Tradition",
    "be obedient": "Conformity-rules", "be self-disciplined": "Conformity-rules", "moderate": "Conformity-rules",
    "be polite": "Conformity-interpersonal", "be honoring parents and elders": "Conformity-interpersonal",
    "be humble": "Humility", "accepting my portion in life": "Humility",
    "be helpful": "Benevolence-caring", "be honest": "Benevolence-caring", "be forgiving": "Benevolence-caring", "true friendship": "Benevolence-caring", "mature love": "Benevolence-caring",
    "be responsible": "Benevolence-dependability", "have loyalty towards friends": "Benevolence-dependability",
    "have equality": "Universalism-concern", "social justice": "Universalism-concern", "have a world at peace": "Universalism-concern", "have a world of peace": "Universalism-concern", "have social justice": "Universalism-concern",
    "be protecting the environment": "Universalism-nature", "have harmony with nature": "Universalism-nature", "have a world of beauty": "Universalism-nature", "have world of beauty": "Universalism-nature", "protecting the environment": "Universalism-nature",
    "be broadminded": "Universalism-tolerance", "have the wisdom to accept others": "Universalism-tolerance", "have wisdom": "Universalism-tolerance",
}

value_item_2_type_10 = {
    "be creative": "Self-direction", "be curious": "Self-direction", "have freedom of thought": "Self-direction",
    "be choosing own goals": "Self-direction", "be independent": "Self-direction", "have freedom of action": "Self-direction", "have privacy": "Self-direction",
    "have an exciting life": "Stimulation", "have a varied life": "Stimulation", "be daring": "Stimulation",
    "have pleasure": "Hedonism", "enjoying life": "Hedonism", "be self-indulgent": "Hedonism", "self-indulgent": "Hedonism",
    "be ambitious": "Achievement", "be successful": "Achievement", "be capable": "Achievement", "be influential": "Achievement", "be intellectual": "Achievement", "self-respect": "Achievement",
    "have authority": "Power", "have social power": "Power", "have wealth": "Power",
    "have a social recognition": "Power", "preserving my public image": "Power", "observing social norms": "Security", "have social recognition": "Power",
    "have a sense of belonging": "Security", "have a good health": "Security", "have no debts": "Security", "be neat and tidy": "Security", "have family security": "Security", "family security": "Security",
    "have a safe country": "Security", "have a stable society": "Security", "have a stable social": "Security",
    "be respecting traditions": "Tradition", "be holding religious faith": "Tradition",
    "be obedient": "Conformity", "be self-disciplined": "Conformity", "moderate": "Conformity",
    "be polite": "Conformity", "be honoring parents and elders": "Conformity",
    "be humble": "Conformity", "accepting my portion in life": "Conformity",
    "be helpful": "Benevolence", "be honest": "Benevolence", "be forgiving": "Benevolence", "true friendship": "Benevolence", "mature love": "Benevolence",
    "be responsible": "Benevolence", "have loyalty towards friends": "Benevolence",
    "have equality": "Universalism", "social justice": "Universalism", "have a world at peace": "Universalism", "have a world of peace": "Universalism", "have social justice": "Universalism",
    "be protecting the environment": "Universalism", "have harmony with nature": "Universalism", "have a world of beauty": "Universalism", "have world of beauty": "Universalism", "protecting the environment": "Universalism",
    "be broadminded": "Universalism", "have the wisdom to accept others": "Universalism", "have wisdom": "Universalism",
}

value_type_19_2_type_10 = {
    "Self-direction-thought": "Self-direction", "Self-direction-action": "Self-direction",
    "Stimulation": "Stimulation", "Hedonism": "Hedonism", "Achievement": "Achievement",
    "Power-dominance": "Power", "Power-resources": "Power", "Face": "Power",
    "Security-personal": "Security", "Security-societal": "Security", "Tradition": "Tradition",
    "Conformity-rules": "Conformity", "Conformity-interpersonal": "Conformity", "Humility": "Conformity",
    "Benevolence-caring": "Benevolence", "Benevolence-dependability": "Benevolence",
    "Universalism-concern": "Universalism", "Universalism-nature": "Universalism", "Universalism-tolerance": "Universalism"
}


def process_annotate_values(value_items, value_types, args, split):
    items, item_labels = [], []
    types, type_labels = [], []
    types_set = set()
    value_type_list = value_type_10_list if args.value_type_dimension == 10 else value_type_19_list
    value_item_2_type = value_item_2_type_10 if args.value_type_dimension == 10 else value_item_2_type_19

    for item_score in value_items:
        try:
            item, score = item_score.split(":")
            item = item[0].upper() + item[1:]
            score = int(score.replace(" ", ""))
        except:
            print("invalid item_score: ", item_score)
            continue
        items.append(item.strip())
        item_labels.append(score)
    
    for type_score in value_types:
        try:
            value_type, score = type_score.split(":")
            value_type = value_type[0].upper() + value_type[1:]
            score = int(score.replace(" ", ""))
        except:
            print("invalid type_score: ", type_score)
            continue
        types.append(value_type.strip())
        type_labels.append(score)
        types_set.add(value_type)
        types_set.add(value_type.rsplit("-", 1)[0])

    if split == "train":
        rel_value_items_num = len(items)
        no_connection_value = random.sample([x for x in value_item_list if x not in items], rel_value_items_num // 2)
        for value in no_connection_value:
            items.append(value)
            label = 0
            item_labels.append(label)
        
        no_connection_type = []
        rel_value_types_num = len(types)
        while(len(no_connection_type) < min(rel_value_items_num // 2, len(value_type_list) - rel_value_items_num)):
            value_type = random.choice(value_type_list)
            if value_type not in types_set and value_type.rsplit("-", 1)[0] not in types_set:
                types_set.add(value_type)
                types_set.add(value_type.rsplit("-", 1)[0])
                no_connection_type.append(value_type)
                types.append(value_type)
                label = 0
                type_labels.append(label)
    else:
        no_connection_value = [x for x in value_item_list if x not in items]
        for value in no_connection_value:
            items.append(value)
            label = 0
            item_labels.append(label)

        for value_type in value_type_list:
            if value_type not in types_set:
                types_set.add(value_type)
                types.append(value_type)
                label = 0
                type_labels.append(label)
    
    if args.value_item_or_type == "value_item":
        return zip(items, item_labels)
    else:
        return zip(types, type_labels)

def prompt_for_classification(dialogue, value, label, args):
    if args.value_item_or_type == "value_item":
        template = "From now on, you are an expert in psychology and sociology. You are familiar with the Schwartz Theory of Basic Values and can correctly identify the values that guide Bob's responses.\n\
        The given value item is {value_definition}.\n\
        The dialogue you need to annotate is:\n\
        {dialogue}\n\
        Please assign one of the {class_num} different labels based on the reflection of the value item in Bob's response.\n\
        The labels are: "

        try:
            value_definition = f"\"{value}\": {value_item_definition_dict[value[0].upper()+value[1:]]}"
        except:
            return None, None
    elif args.value_item_or_type == "value_type":
        template = "From now on, you are an expert in psychology and sociology. You are familiar with the Schwartz Theory of Basic Values and can correctly identify the values that guide Bob's responses.\n\
        The given value type is {value_definition}.\n\
        The dialogue you need to annotate is:\n\
        {dialogue}\n\
        Please assign one of the {class_num} different labels based on the reflection of the value item in Bob's response.\n\
        The labels are: "

        try:
            if args.value_type_dimension == 10:
                value_type_definition_dict = value_type_10_definition_dict
            elif args.value_type_dimension == 19:
                value_type_definition_dict = value_type_19_definition_dict
            value_definition = f"\"{value}\": {value_type_definition_dict[value[0].upper()+value[1:]]}"
        except:
            return None, None

    if args.class_num == 3:
        template += "\"Opposed\", \"No connection\", \"Important\"."
    elif args.class_num == 4:
        template += "\"Opposed\", \"No connection\", \"Relevant, not a major focus\", \"Important\"."
    prompt = template.format(value_definition = value_definition, dialogue = dialogue, class_num = args.class_num)

    if args.model_type == "classification":
        label = int(label) + 1
    elif args.model_type == "regression":
        label = int(label)
    
    return prompt, label

def get_value_id(value, args):
    if args.value_item_or_type == "value_item":
        if value == "Have social justice":
            return value_item_list.index("Social justice")
        if value == "Have social recognition":
            return value_item_list.index("Have a social recognition")
        if value == "Have security":
            return value_item_list.index("Have family security")
        return value_item_list.index(value)
    else:
        if args.value_type_dimension == 10:
            value_type_list = value_type_10_list
        else:
            value_type_list = value_type_19_list
        return value_type_list.index(value)

def get_type_10_id(id, args):
    if args.value_item_or_type == "value_item":
        return value_type_10_list.index(value_item_2_type_10[value_item_list[id].lower()])
    if args.value_item_or_type == "value_type":
        if args.value_type_dimension == 10:
            return id
        else:
            return value_type_10_list.index(value_type_19_2_type_10[value_type_19_list[id]])


##############################################

def load_saferlhf_data():  # avg length: 75
    data_path = "/home/jingyao/projects/Alignment/Value_Benchmark/local_data/hh-value/saferlhf"
    train_data = []
    with open(os.path.join(data_path, "train_safety.jsonl"), "r") as f:
        for line in tqdm(f, desc="Loading train data"):
            data = json.loads(line)
            prompt = data["prompt"]
            response_0, response_1 = data["response_0"], data["response_1"]
            is_response_0_safe, is_response_1_safe = data["is_response_0_safe"], data["is_response_1_safe"]
            safer_response_id = data["safer_response_id"]
            if safer_response_id == 0:
                category = [k for k, v in data["category_1"].items() if v]
                chosen = f"Human: {prompt}\nAssistant: {response_0}"
                rejected = f"Human: {prompt}\nAssistant: {response_1}"
            else:
                category = [k for k, v in data["category_0"].items() if v]
                chosen = f"Human: {prompt}\nAssistant: {response_1}"
                rejected = f"Human: {prompt}\nAssistant: {response_0}"

            train_data.append({"chosen": chosen, "rejected": rejected, "safety_issues": category})

    test_data = []
    with open(os.path.join(data_path, "test_safety.jsonl"), "r") as f:
        for line in tqdm(f, desc="Loading test data"):
            data = json.loads(line)
            prompt = data["prompt"]
            response_0, response_1 = data["response_0"], data["response_1"]
            is_response_0_safe, is_response_1_safe = data["is_response_0_safe"], data["is_response_1_safe"]
            safer_response_id = data["safer_response_id"]
            if safer_response_id == 0:
                chosen = f"Human: {prompt}\nAssistant: {response_0}"
                rejected = f"Human: {prompt}\nAssistant: {response_1}"
            else:
                chosen = f"Human: {prompt}\nAssistant: {response_1}"
                rejected = f"Human: {prompt}\nAssistant: {response_0}"
            test_data.append({"chosen": chosen, "rejected": rejected})

    return data_path, train_data, test_data