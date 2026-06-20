import re
import json
import copy
import random
import argparse
import numpy as np
from tqdm import tqdm
from rouge_score import rouge_scorer
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import *
from models import *

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=int, default=1)
    parser.add_argument('--country', type=str, default='China')
    parser.add_argument('--output_dir', type=str, default='../data')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--embed_model', type=str, default='text-embedding-3-small')
    parser.add_argument('--question_generator', type=str, default='gpt-4')
    parser.add_argument('--main_generator', type=str, default='gpt-4')
    parser.add_argument('--multi_generators', nargs='+', default=None, help="List of generators to use, e.g., --multi_generators gpt-4 gpt-3.5")
    parser.add_argument('--compare_countries', nargs='+', default=None, help="List of countries to compare with, e.g., --compare_countries United_States Germany Japan")
    return parser.parse_args()

def parse_gpt_questions(text):
    pattern = r"\[question \d+\]: (.+?) \((.+? question)\)"
    matches = re.findall(pattern, text)
    parsed_questions = [{"question": q.strip(), "type": q_type.strip()} for q, q_type in matches]
    return parsed_questions

def parse_cultural_points(text):
    client = create_aoai_client()
    points = []
    for point in text.split("\n"):
        # point should start with a number, e.g., "1. Point description"
        point = point.strip()
        if not point or not re.match(r"^\d+\.", point):
            continue
        point = point.split(".", 1)[-1].strip()  # remove the number prefix
        points.append(point)
    if not points:
        return []
    # one batched embedding call instead of one per point
    resp = client.embeddings.create(input=points, model=args.embed_model)
    return [{"point": p, "embed": d.embedding} for p, d in zip(points, resp.data)]

def compute_cosine_similarity(vec_array, vec):
    """
    Compute cosine similarity between a vector and an array of vectors.
    :param vec_array: np.ndarray, shape (n_samples, n_features)
    :param vec: np.ndarray, shape (n_features,)
    :return: list of cosine similarities
    """
    from sklearn.metrics.pairwise import cosine_similarity
    return cosine_similarity(np.array(vec_array), np.array(vec).reshape(1, -1)).flatten().tolist()

def matrix_cosine_similarity(A, B):
    A_norm = A / np.linalg.norm(A, axis=1, keepdims=True)
    B_norm = B / np.linalg.norm(B, axis=1, keepdims=True)
    similarity = np.dot(A_norm, B_norm.T)
    return similarity # shape: (A_rows, B_rows)

def majority_consensus(raw_consensus_matrix):
    final_consensus = {}
    consensus_scores = np.sum(raw_consensus_matrix, axis=0) / float(raw_consensus_matrix.shape[0])
    for i in range(raw_consensus_matrix.shape[1]):
        final_consensus[i] = float(consensus_scores[i])
    return final_consensus

def init_rep_dist_questions(args):
    question_generator = create_model(args.question_generator, args)
    client = create_aoai_client()
    all_topics = [k for k in wvs_topics] # a total of value-related topics
    os.makedirs(os.path.join(args.output_dir, args.country), exist_ok=True)

    if os.path.exists(os.path.join(args.output_dir, args.country, f"rep_dist_questions_{args.question_generator.split('/')[-1]}.json")):
        topic_questions = json.load(open(os.path.join(args.output_dir, args.country, f"rep_dist_questions_{args.question_generator.split('/')[-1]}.json"), "r"))
    else:
        topic_questions = defaultdict(list)
        for topic, examples in tqdm(wvs_topics_examples.items(), desc="Initializing topics with examples"):
            examples = [{'question': q, 'embedding': client.embeddings.create(input = [q], model=args.embed_model).data[0].embedding} for q in examples]
            topic_questions[topic] = examples[:]

    max_questions_per_topic = 30
    input_words, output_words = 0, 0  # total: 800 words * 1.5 = 1200 tokens
    with tqdm(total=1000, desc=f"Generating questions for {args.country} cultural values") as pbar:
        for r_idx in range(1000):
            topic = random.choice(all_topics)
            if len(topic_questions[topic]) >= max_questions_per_topic:
                continue
            definition = wvs_topics[topic]
            if not definition.lower().startswith(topic.lower().split()[0]):
                definition = f"{topic}: {definition}"
            selected_examples = random.sample(topic_questions[topic], 2)

            prompt_base = represent_distinctive_question_generate_prompt.format(topic = definition, country = args.country, example_1 = selected_examples[0]['question'], example_2 = selected_examples[1]['question'])

            output = question_generator.get_top_p_answer(prompt_base, "", max_tokens = 512, temperature = 0.7, top_p = 0.8)

            input_words += len(prompt_base.split())
            output_words += len(output.split())
            pbar.update(1)
            pbar.set_description(f"Generating questions for {args.country} on {topic}, input words: {input_words}, output words: {output_words}")
        
            questions = parse_gpt_questions(output)
            existing_questions = topic_questions[topic]
            existing_question_embeds = [q["embedding"] for q in existing_questions]
            repeat_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            for question in questions:
                rouge_scores = [repeat_scorer.score(existing_questions[idx]["question"], question["question"])['rougeL'].fmeasure for idx in range(len(existing_questions))]
                if max(rouge_scores) > 0.6:
                    print("Questions with high similarity. Rouge scores: ", max(rouge_scores))
                    continue

                question["embedding"] = client.embeddings.create(input = [question["question"]], model=args.embed_model).data[0].embedding
                cosine_scores = compute_cosine_similarity(existing_question_embeds, question["embedding"])
                if max(cosine_scores) > 0.9:
                    print("Questions with high cosine similarity. Cosine scores: ", max(cosine_scores))
                    print("Question: ", question["question"])
                    continue
                
                most_similar = existing_questions[rouge_scores.index(max(rouge_scores))]
                question["most_similar"] = most_similar["question"]
                question["topic"] = topic
                topic_questions[topic].append(question)
            
            if len(topic_questions[topic]) >= max_questions_per_topic:
                all_topics.remove(topic)
                if len(all_topics) == 0:
                    break
            
            if r_idx % 50 == 0:
                print("Saving questions...")

                with open(os.path.join(args.output_dir, args.country, f"rep_dist_questions_{args.question_generator.split('/')[-1]}.json"), "w") as f:
                    json.dump(topic_questions, f, indent=4)

    with open(os.path.join(args.output_dir, args.country, f"rep_dist_questions_{args.question_generator.split('/')[-1]}.json"), "w") as f:
        json.dump(topic_questions, f, indent=4)

def rep_dist_optimization(args):
    # generate represent and distinct answers --> refine questions --> generate represent and distinct answers, iteratively
    
    generator = create_model(args.main_generator, args)
    topic_questions = json.load(open(os.path.join(args.output_dir, args.country, f"rep_dist_questions_{args.question_generator.split('/')[-1]}.json"), "r"))
    questions = []
    for topic, topic_question in topic_questions.items():
        for q in topic_question:
            q["topic"] = topic
        questions.extend(topic_question)

    # if args.country != "Japan":
    random.seed(42)
    random.shuffle(questions)  # set a fixed random seed for reproducibility, and shuffle the questions to ensure a good mix of topics in each batch

    existing_answers = []
    if os.path.exists(os.path.join(args.output_dir, args.country, f"rep_dist_answers_{args.main_generator.split('/')[-1]}.jsonl")):
        existing_answers = open(os.path.join(args.output_dir, args.country, f"rep_dist_answers_{args.main_generator.split('/')[-1]}.jsonl"), "r").readlines()
    fw = open(os.path.join(args.output_dir, args.country, f"rep_dist_answers_{args.main_generator.split('/')[-1]}.jsonl"), "w")

    input_words, output_words = 0, 0
    pbar = tqdm(questions, desc=f"Representativeness & distinctiveness optimization for {questions[0]["topic"]} for {args.country}")

    for start_qidx in range(0, len(questions), args.batch_size):
        if start_qidx < len(existing_answers):
            for idx in range(start_qidx, min(start_qidx + args.batch_size, len(existing_answers))):
                fw.write(existing_answers[idx])
            pbar.update(args.batch_size)
            continue
        question_batch = questions[start_qidx:start_qidx + args.batch_size]
        
        for round_idx in range(1, 3):  # only do 2 rounds of optimization
            for question in question_batch:
                question[f"round_{round_idx}_content"] = {}

            ## generate answers to maximize log[p(y|x)]
            print("generating representative and distinctive answers...")
            input_w1, output_w1 = generate_rep_dist_answers(args, question_batch, round_idx, generator)
            print("computing answer consensus...")
            input_w2, output_w2 = compute_answer_consensus(args, question_batch, round_idx, generator)
            print("computing answer distinctiveness...")
            input_w3, output_w3 = compute_answer_distinctiveness(args, question_batch, round_idx, generator)
            print("integrating representative and distinctive answers...")
            input_w4, output_w4 = integrate_rep_dist_answers(args, question_batch, round_idx, generator)
            ## refine the question to maximize log[p(x)] + log[p(y|x)]
            print("refining questions to maximize log[p(x)] + log[p(y|x)]...")
            input_w5, output_w5 = refine_rep_dist_question(args, question_batch, round_idx, generator)
            input_words += (input_w1 + input_w2 + input_w3 + input_w4 + input_w5)
            output_words += (output_w1 + output_w2 + output_w3 + output_w4 + output_w5)

        pbar.update(args.batch_size)
        pbar.set_description(f"Representativeness & distinctiveness optimization for {question_batch[0]['topic']} for {args.country}, input words: {input_words}, output words: {output_words}")
        for question in question_batch:
            question["country"] = args.country
            question["round_idx"] = round_idx
            fw.write(json.dumps(question) + "\n")
    fw.close()

def generate_rep_dist_answers(args, question_batch, round_idx, generator):
    input_words, output_words = 0, 0
    other_countries = [None] if args.compare_countries is None else args.compare_countries

    prompts_base = []
    if args.compare_countries is not None:
        for question in question_batch:
            question_str = question["question"] if round_idx == 1 else question[f"round_{round_idx - 1}_content"]["refined_question"]
            for cidx, country in enumerate(other_countries):
                prompt_base = represent_answer_expert_role.format(country = country, role = "cross-cultural researcher", question = question_str)
                prompts_base.append(prompt_base)
                input_words += len(prompt_base.split())
        responses = generator.get_batch_top_p_answer(prompts_base, [""] * len(prompts_base), max_tokens = 512, temperature = 0.7, top_p = 0.8)
        for qidx, question in enumerate(question_batch):
            question[f"round_{round_idx}_content"]["other_country_responses"] = []
            for cidx, country in enumerate(other_countries):
                output_words += len(responses[cidx + qidx * len(other_countries)].split())
                question[f"round_{round_idx}_content"]["other_country_responses"].append({"response": responses[cidx + qidx * len(other_countries)], "country": country})
    else:
        for question in question_batch:
            question_str = question["question"] if round_idx == 1 else question[f"round_{round_idx - 1}_content"]["refined_question"]
            prompt_base = no_culture_answer.format(question = question_str)
            prompts_base.append(prompt_base)
            input_words += len(prompt_base.split())
        responses = generator.get_batch_top_p_answer(prompts_base, [""] * len(prompts_base), max_tokens = 512, temperature = 0.7, top_p = 0.8)
        for qidx, question in enumerate(question_batch):
            output_words += len(responses[qidx].split())
            question[f"round_{round_idx}_content"]["other_country_responses"] = [{"response": responses[qidx], "country": "Universal"}]

    culture_profiles = json.load(open(os.path.join(args.output_dir, "WVS_demographics.json"), "r"))[args.country]
    culture_profiles = random.sample(culture_profiles, 2)  # select 2 profiles for demographic responses
    cultural_experts = random.sample(expert_roles, 2)  # use the last 2 experts as cultural experts

    prompts_base = []
    ## generate answers from demographic profiles
    for question in question_batch:
        question_str = question["question"] if round_idx == 1 else question[f"round_{round_idx - 1}_content"]["refined_question"]
        other_response = "\n".join(f"{x['country']}: {x['response']}" for x in question[f"round_{round_idx}_content"]["other_country_responses"])

        for pidx, profile in enumerate(culture_profiles):
            demographic = demographics.format(gender = profile['sex'], age = profile['age'], education = profile['education'], occupation = profile['occupation'], social_class = profile['social_class'], income_level = profile['income_level'])
            prompt_base = rep_dist_answer_demographic_role.format(country = args.country, demographic_info = demographic, question = question_str, other_response = other_response)
            prompts_base.append(prompt_base)
            input_words += len(prompt_base.split())        
        ## generate answers from cultural experts
        for cidx, expert in enumerate(cultural_experts):
            prompt_base = rep_dist_answer_expert_role.format(country = args.country, role = expert, question = question_str, other_response = other_response)
            prompts_base.append(prompt_base)
            input_words += len(prompt_base.split())

    responses = generator.get_batch_top_p_answer(prompts_base, [""] * len(prompts_base), max_tokens = 512, temperature = 0.7, top_p = 0.8)

    prompts_base = []
    one_response_count = len(responses) // len(question_batch)
    for qidx, question in enumerate(question_batch):
        question_str = question["question"] if round_idx == 1 else question[f"round_{round_idx - 1}_content"]["refined_question"]
        one_responses = responses[qidx * one_response_count:(qidx + 1) * one_response_count]
        for cidx, response in enumerate(one_responses):
            output_words += len(response.split())
            question[f"round_{round_idx}_content"][f"response_{cidx}"] = response
            prompt_base = point_extraction_prompt.format(question= question_str, response=response)
            input_words += len(prompt_base.split())
            prompts_base.append(prompt_base)

        ## also extract points from other countries' responses
        for other_resp in question[f"round_{round_idx}_content"]["other_country_responses"]:
            prompt_base = point_extraction_prompt.format(question= question_str, response=other_resp["response"])
            input_words += len(prompt_base.split())
            prompts_base.append(prompt_base)

    responses = generator.get_batch_top_p_answer(prompts_base, [""] * len(prompts_base), max_tokens=256, temperature=0.7, top_p=0.8)

    other_response_count = len(other_countries) if args.compare_countries is not None else 1
    for qidx, question in enumerate(question_batch):
        one_responses = responses[qidx * (one_response_count + other_response_count):(qidx + 1) * (one_response_count + other_response_count)]
        for cidx, response in enumerate(one_responses[:one_response_count]):  # exclude the last response which is the main country response
            output_words += len(response.split())
            question[f"round_{round_idx}_content"][f"response_{cidx}_cultural_points"] = parse_cultural_points(response.strip())
        
        for cidx, response in enumerate(one_responses[one_response_count:]):
            output_words += len(response.split())
            question[f"round_{round_idx}_content"]["other_country_responses"][cidx]["cultural_points"] = parse_cultural_points(response.strip())

    return input_words, output_words

def compute_answer_consensus(args, question_batch, round_idx, generator):
    input_words, output_words = 0, 0

    ## compute the consensus score for each question
    country_list = ["United States", "United Kingdom", "Germany", "Italy", "South Korea", "China", "Japan", "India", "Singapore", "Indonesia", "Russia", "Poland"]
    if args.country.replace("_", " ") in country_list:
        country_list.remove(args.country.replace("_", " "))

    culture_profiles = random.sample(json.load(open(os.path.join(args.output_dir, "WVS_demographics.json"), "r"))[args.country], 5) # [:5]
    cultural_experts = random.sample(expert_roles, 5)  # use 5 experts as cultural experts
    foreign_experts = random.sample(country_list, 2)
    pattern = re.compile(r"Point\s*(\d+)\s*:\s*([1-5])\b", flags=re.IGNORECASE)
    question_labels = []

    prompts_base = []
    for question in question_batch:
        question_str = question["question"] if round_idx == 1 else question[f"round_{round_idx - 1}_content"]["refined_question"]
        all_cultural_points = []
        all_point_embeds = []
        for cidx in range(8):
            if "response_{}_cultural_points".format(cidx) in question[f"round_{round_idx}_content"]:
                cultural_points = question[f"round_{round_idx}_content"]["response_{}_cultural_points".format(cidx)]
                all_cultural_points.extend([x["point"] for x in cultural_points])
                all_point_embeds.extend([x["embed"] for x in cultural_points])
    
        X = np.array(all_point_embeds)
        if X.shape[0] < 2:
            continue
        hac = AgglomerativeClustering(distance_threshold=0.3, linkage='average', metric='cosine', n_clusters=None)
        hac.fit(X)
        labels = hac.labels_
        question_labels.append(labels)

        consensus_points_str = ""
        consensus_point_embeds = []
        for i in range(len(set(labels))):
            cluster_idx = np.where(labels == i)[0]  # get the indices of points in this cluster
            cluster_points = [all_cultural_points[idx] for idx in cluster_idx]  # get the cultural points in this cluster
            cluster_embeds = X[cluster_idx]  # get the embeddings of points in this cluster
            centroid = np.mean(cluster_embeds, axis=0, keepdims=False)
            consensus_point_embeds.append(centroid.tolist())  # compute the mean embedding for the cluster
            point_str = ", ".join([x.strip() for x in cluster_points])
            consensus_points_str += f"Point {i + 1}: [{point_str}]\n"
        question[f"round_{round_idx}_content"]["consensus_point_str"] = consensus_points_str
        question[f"round_{round_idx}_content"]["consensus_point_embeds"] = consensus_point_embeds
    
        for cidx, profile in enumerate(culture_profiles):
            demographic = demographics.format(gender = profile['sex'], age = profile['age'], education = profile['education'], occupation = profile['occupation'], social_class = profile['social_class'], income_level = profile['income_level'])

            prompt_base = represent_point_judgment_demographic.format(country = args.country, demographic_info = demographic, question = question_str, cultural_points = consensus_points_str)
            prompts_base.append(prompt_base)

        for cidx, expert in enumerate(cultural_experts):
            prompt_base = represent_point_judgment_expert.format(country = args.country, role = expert, question = question_str, cultural_points = consensus_points_str)
            prompts_base.append(prompt_base)
            
        for cidx, foreign in enumerate(foreign_experts):
            prompt_base = represent_point_judgment_foreign.format(foreign = foreign, country = args.country, question = question_str, cultural_points = consensus_points_str)
            prompts_base.append(prompt_base)

    responses = []  # [responses for question 1, responses for question 2, ...]
    max_batch_size = min(64, args.batch_size * 4)
    for start in range(0, len(prompts_base), max_batch_size):
        prompt_base_batch = prompts_base[start:start + max_batch_size]
        responses.extend(generator.get_batch_top_p_answer(prompt_base_batch, [""] * len(prompt_base_batch), max_tokens = 512, temperature = 0.7, top_p = 0.8))
    
    for qidx, question in enumerate(question_batch):
        labels = question_labels[qidx]
        one_responses_count = len(culture_profiles) + len(cultural_experts) + len(foreign_experts)
        raw_consensus_matrix = np.zeros((len(culture_profiles) + len(cultural_experts) + len(foreign_experts), len(set(labels))))
        one_responses = responses[qidx * one_responses_count:(qidx + 1) * one_responses_count]

        for cidx, response in enumerate(one_responses):
            input_words += len(prompts_base[qidx * one_responses_count + cidx].split())
            output_words += len(response.split())
            for match in pattern.finditer(response):
                point_idx = int(match.group(1)) - 1
                score = int(match.group(2))
                # judgment = 1 if score >= 4 else 0
                if point_idx >= raw_consensus_matrix.shape[1]:
                    break
                raw_consensus_matrix[cidx, point_idx] = score

        likert_consensus_matrix = raw_consensus_matrix.copy()
        raw_consensus_matrix[raw_consensus_matrix < 4] = 0  # convert to binary matrix, 0 or 1
        raw_consensus_matrix[raw_consensus_matrix >= 4] = 1  #
        final_consensus = majority_consensus(raw_consensus_matrix)

        question[f"round_{round_idx}_content"]["likert_consensus"] = likert_consensus_matrix.tolist()
        question[f"round_{round_idx}_content"]["consensus_scores"] = final_consensus
    return input_words, output_words

def compute_answer_distinctiveness(args, question_batch, round_idx, generator):
    ## compute the distinctiveness score for cultural points under each question
    input_words, output_words = 0, 0

    for qidx, question in enumerate(question_batch):
        consensus_point_embeds = np.array(question[f"round_{round_idx}_content"]["consensus_point_embeds"]) # (num_points, embed_dim)
        # print("Consensus point embeds shape: ", consensus_point_embeds.shape)

        all_other_sims = []
        for other_resp in question[f"round_{round_idx}_content"]["other_country_responses"]:
            other_resp_point_embeds = np.array([cp["embed"] for cp in other_resp["cultural_points"]])
            if other_resp_point_embeds.shape[0] == 0:
                max_cosine_similarity = 0.8 * np.ones(consensus_point_embeds.shape[0])  # if there is no point in the other response, set similarity to 1 (not distinctive at all)
            else:
                cosine_similarity = matrix_cosine_similarity(consensus_point_embeds, other_resp_point_embeds)  # shape: (num_consensus_points, num_other_points)
                max_cosine_similarity = np.max(cosine_similarity, axis=1)  # shape: (num_consensus_points,)
            all_other_sims.append(max_cosine_similarity)
        all_other_sims = np.array(all_other_sims)  # shape: (num_other_countries, num_consensus_points)
        distinctiveness_score = 1 - np.mean(all_other_sims, axis=0)  # shape: (num_consensus_points,)
        final_distinctiveness = {}
        for pidx in range(consensus_point_embeds.shape[0]):
            final_distinctiveness[pidx] = round(float(distinctiveness_score[pidx]), 3)
        question[f"round_{round_idx}_content"]["distinctiveness_scores"] = final_distinctiveness
    return input_words, output_words

def integrate_rep_dist_answers(args, question_batch, round_idx, generator):
    input_words, output_words = 0, 0

    prompts_base = []
    for question in question_batch:
        question_str = question["question"] if round_idx == 1 else question[f"round_{round_idx - 1}_content"]["refined_question"]

        consensus_points_str = ""
        consensus_scores = question[f"round_{round_idx}_content"]["consensus_scores"]
        distinctiveness_scores = question[f"round_{round_idx}_content"]["distinctiveness_scores"]
        consensus_points = question[f"round_{round_idx}_content"]["consensus_point_str"].strip().split("\n")
        count = 0
        # print("Consensus scores: ", consensus_scores)
        for idx in range(len(consensus_scores)):
            # rep_score = consensus_scores[idx]
            # dist_score = distinctiveness_scores[idx] 
            # dist_score = dist_score * np.log(dist_score / (1 - dist_score)) + np.log(1 - dist_score)  # convert distinctiveness to log-odds score
            if int(idx) not in consensus_scores or consensus_scores[int(idx)] < 0.75 or distinctiveness_scores[int(idx)] < 0.25:  # only keep the points with high consensus and distinctiveness
                continue
            count += 1
            consensus_points_str += f"Point {count}: {consensus_points[idx].split(": ", 1)[-1].strip()}\n"
        # print("Consensus points: ", consensus_points_str)

        prompt_base = rep_dist_aggregation_prompt.format(country=args.country, question=question_str, consensus_points=consensus_points_str)
        input_words += len(prompt_base.split())
        prompts_base.append(prompt_base)
    responses = generator.get_batch_top_p_answer(prompts_base, [""] * len(prompts_base), max_tokens=1024, temperature=0.7, top_p=0.8)
    for question, response in zip(question_batch, responses):
        output_words += len(response.split())
        question[f"round_{round_idx}_content"]["integrated_answer"] = response.strip()
    return input_words, output_words

def refine_rep_dist_question(args, question_batch, round_idx, refiner):
    input_words, output_words = 0, 0

    prompts_base = []
    for qidx, question in enumerate(question_batch):
        topic = question["topic"]
        question_str = question["question"] if round_idx == 1 else question[f"round_{round_idx - 1}_content"]["refined_question"]

        consensus_points = question[f"round_{round_idx}_content"]["consensus_point_str"].strip().split("\n")
        representativeness_scores = question[f"round_{round_idx}_content"]["consensus_scores"]
        distinctiveness_scores = question[f"round_{round_idx}_content"]["distinctiveness_scores"]

        consensus_score_str = ""
        for idx, point in enumerate(consensus_points):
            represent_reward = round(representativeness_scores[int(idx)], 2)
            distinct_reward = round(distinctiveness_scores[int(idx)], 2)
            consensus_score_str += f"{point} (representativeness reward: {represent_reward}, distinctiveness reward: {distinct_reward})\n"

        other_responses = "\n".join(f"{x['country']}: {x['response']}" for x in question[f"round_{round_idx}_content"]["other_country_responses"])

        prompt_base = rep_dist_reflection_prompt.format(country=args.country, topic=topic, question=question_str, cultural_points=consensus_score_str, other_response=other_responses)
        input_words += len(prompt_base.split())
        prompts_base.append(prompt_base)

    responses = refiner.get_batch_top_p_answer(prompts_base, [""] * len(prompts_base), max_tokens=512, temperature=0.7, top_p=0.8)
    messages = []
    for prompt_base, response in zip(prompts_base, responses):
        output_words += len(response.split())
        input_words += len(prompt_base.split() + rep_dist_refine_prompt.split())
        messages.append([{"role": "user", "content": prompt_base},
                {"role": "assistant", "content": response},
                {"role": "user", "content": rep_dist_refine_prompt.format(country=args.country)}])
    responses = refiner.get_batch_message_answer(messages, max_tokens=128, temperature=0.7, top_p=0.8)
    for idx, response in enumerate(responses):
        output_words += len(response.split())
        question_batch[idx][f"round_{round_idx}_content"]["refined_question"] = response.split(":", 1)[-1].strip()
    return input_words, output_words

def rep_dist_optimization_multigenerator(args):
    generators = [create_model(model_name, args) for model_name in args.multi_generators]
    topic_questions = json.load(open(os.path.join(args.output_dir, args.country, f"rep_dist_questions_{args.question_generator.split('/')[-1]}.json"), "r"))
    questions = []
    for topic, topic_question in topic_questions.items():
        for q in topic_question:
            q["topic"] = topic
        questions.extend(topic_question)
    
    existing_answers = []
    multi_generators_str = "_".join([g.split("/")[-1] for g in args.multi_generators])
    if os.path.exists(os.path.join(args.output_dir, args.country, f"rep_dist_answers_{multi_generators_str}.jsonl")):
        existing_answers = open(os.path.join(args.output_dir, args.country, f"rep_dist_answers_{multi_generators_str}.jsonl"), "r").readlines()
    fw = open(os.path.join(args.output_dir, args.country, f"rep_dist_answers_{multi_generators_str}.jsonl"), "w")

    input_words, output_words = 0, 0
    pbar = tqdm(questions, desc=f"Representativeness & distinctiveness optimization for {questions[0]['topic']} for {args.country}")

    for start_qidx in range(0, len(questions), args.batch_size):
        if start_qidx < len(existing_answers):
            for idx in range(start_qidx, min(start_qidx + args.batch_size, len(existing_answers))):
                fw.write(existing_answers[idx])
            pbar.update(args.batch_size)
            continue
        question_batch = questions[start_qidx:start_qidx + args.batch_size]
        
        for round_idx in range(1, 3):  # only do 2 rounds of optimization
            for question in question_batch:
                question[f"round_{round_idx}_content"] = {}

            ## generate answers to maximize log[p(y|x)]
            print("generating representative and distinctive answers...")
            input_w1, output_w1 = generate_rep_dist_answers_multigenerator(args, question_batch, round_idx, generators)
            print("computing answer consensus...")
            input_w2, output_w2 = compute_answer_consensus_multigenerator(args, question_batch, round_idx, generators)
            print("computing answer distinctiveness...")
            input_w3, output_w3 = compute_answer_distinctiveness(args, question_batch, round_idx, generators[0])
            print("integrating representative and distinctive answers...")
            input_w4, output_w4 = integrate_rep_dist_answers(args, question_batch, round_idx, generators[0])
            ## refine the question to maximize log[p(x)] + log[p(y|x)]
            print("refining questions to maximize log[p(x)] + log[p(y|x)]...")
            input_w5, output_w5 = refine_rep_dist_question(args, question_batch, round_idx, generators[0])
            input_words += (input_w1 + input_w2 + input_w3 + input_w4 + input_w5)
            output_words += (output_w1 + output_w2 + output_w3 + output_w4 + output_w5)

        pbar.update(args.batch_size)
        pbar.set_description(f"Representativeness & distinctiveness optimization for {question_batch[0]['topic']} for {args.country}, input words: {input_words}, output words: {output_words}")
        for question in question_batch:
            question["country"] = args.country
            question["round_idx"] = round_idx
            fw.write(json.dumps(question) + "\n")
    fw.close()

def generate_rep_dist_answers_multigenerator(args, question_batch, round_idx, generators):
    input_words, output_words = 0, 0

    target = args.country.replace("_", " ")
    country_group = country_2_group.get(args.country, "Unknown")
    non_country_groups = [x for x in country_groups if x != country_group]

    similar_countries = [x for x in group_countries[country_group] if x != target]
    different_group_countries = [y for x in non_country_groups for y in group_countries[x] if y != target]

    n_similar = min(1, len(similar_countries))
    n_different = min(4, len(different_group_countries))
    other_countries = random.sample(similar_countries, n_similar) + random.sample(different_group_countries, n_different)

    # other_countries = [None] if args.compare_countries is None else args.compare_countries

    prompts_base = []
    if other_countries is not None:
        for question in question_batch:
            question_str = question["question"] if round_idx == 1 else question[f"round_{round_idx - 1}_content"]["refined_question"]
            for cidx, country in enumerate(other_countries):
                prompt_base = represent_answer_expert_role.format(country = country, role = "cross-cultural researcher", question = question_str)
                prompts_base.append(prompt_base)
                input_words += len(prompt_base.split())
        responses = generators[0].get_batch_top_p_answer(prompts_base, [""] * len(prompts_base), max_tokens = 512, temperature = 0.7, top_p = 0.8)
        for qidx, question in enumerate(question_batch):
            question[f"round_{round_idx}_content"]["other_country_responses"] = []
            for cidx, country in enumerate(other_countries):
                output_words += len(responses[cidx + qidx * len(other_countries)].split())
                question[f"round_{round_idx}_content"]["other_country_responses"].append({"response": responses[cidx + qidx * len(other_countries)], "country": country})
    else:
        for question in question_batch:
            question_str = question["question"] if round_idx == 1 else question[f"round_{round_idx - 1}_content"]["refined_question"]
            prompt_base = no_culture_answer.format(question = question_str)
            prompts_base.append(prompt_base)
            input_words += len(prompt_base.split())
        responses = generators[0].get_batch_top_p_answer(prompts_base, [""] * len(prompts_base), max_tokens = 512, temperature = 0.7, top_p = 0.8)
        for qidx, question in enumerate(question_batch):
            output_words += len(responses[0].split())
            question[f"round_{round_idx}_content"]["other_country_responses"] = [{"response": responses[qidx], "country": "Universal"}]

    culture_expert = "cross-cultural researcher"

    prompts_base = []
    ## generate answers from demographic profiles
    for question in question_batch:
        question_str = question["question"] if round_idx == 1 else question[f"round_{round_idx - 1}_content"]["refined_question"]
        other_response = "\n".join(f"{x['country']}: {x['response']}" for x in question[f"round_{round_idx}_content"]["other_country_responses"])

        prompt_base = rep_dist_answer_expert_role.format(country = args.country, role = culture_expert, question = question_str, other_response = other_response)
        prompts_base.append(prompt_base)
        input_words += len(prompt_base.split())
    
    responses = []  # [res1, res2, .., res1, res2, ...] where res1 and res2 are answers from two different generators
    # parallelize across generators (IO-bound LLM calls)
    def _gen_target_answers(gen):
        return gen.get_batch_top_p_answer(prompts_base, [""] * len(prompts_base), max_tokens=512, temperature=0.7, top_p=0.8)
    with ThreadPoolExecutor(max_workers=len(generators)) as ex:
        per_gen_responses = list(ex.map(_gen_target_answers, generators))
    for chunk in per_gen_responses:
        responses.extend(chunk)

    prompts_base = []  # [prompt1 for res1, prompt2 for res2, ..., prompt1 for res1, prompt2 for res2, ...]
    for gix, generator in enumerate(generators):
        for qidx, question in enumerate(question_batch):
            question_str = question["question"] if round_idx == 1 else question[f"round_{round_idx - 1}_content"]["refined_question"]
            question[f"round_{round_idx}_content"][f"response_{gix}"] = responses[gix * len(question_batch) + qidx]
            prompt_base = point_extraction_prompt.format(question= question_str, response=responses[gix * len(question_batch) + qidx])
            input_words += len(prompt_base.split())
            prompts_base.append(prompt_base)

            for other_resp in question[f"round_{round_idx}_content"]["other_country_responses"]:
                prompt_base = point_extraction_prompt.format(question= question_str, response=other_resp["response"])
                input_words += len(prompt_base.split())
                prompts_base.append(prompt_base)

    responses = generators[0].get_batch_top_p_answer(prompts_base, [""] * len(prompts_base), max_tokens=256, temperature=0.7, top_p=0.8) # [prompts for q1 (target + other country responses), prompts for q2, ..., prompts for q1, prompts for q2, ...] where the first half is for the first generator and the second half is for the second generator

    other_response_count = len(other_countries) if other_countries is not None else 1
    ## (1 + other_response_count) * len(question_batch) * len(generators) responses in total
    for gix, generator in enumerate(generators):
        for qidx, question in enumerate(question_batch):
            one_question_responses = responses[gix * ((1 + other_response_count) * len(question_batch)) + qidx * (1 + other_response_count) : gix * ((1 + other_response_count) * len(question_batch)) + (qidx + 1) * (1 + other_response_count)]
            output_words += len(one_question_responses[0].split())
            question[f"round_{round_idx}_content"][f"response_{gix}_cultural_points"] = parse_cultural_points(one_question_responses[0].strip())
            
            if gix == 0:
                for oidx, other_resp in enumerate(question[f"round_{round_idx}_content"]["other_country_responses"]):
                    other_response = one_question_responses[oidx + 1]
                    output_words += len(other_response.split())
                    question[f"round_{round_idx}_content"]["other_country_responses"][oidx]["cultural_points"] = parse_cultural_points(other_response.strip())
    return input_words, output_words

def compute_answer_consensus_multigenerator(args, question_batch, round_idx, generators):
    input_words, output_words = 0, 0

    ## compute the consensus score for each question
    country_list = ["United States", "United Kingdom", "Germany", "Italy", "South Korea", "China", "Japan", "India", "Singapore", "Indonesia", "Russia", "Poland"]
    if args.country.replace("_", " ") in country_list:
        country_list.remove(args.country.replace("_", " "))

    # culture_profiles = random.sample(json.load(open(os.path.join(args.output_dir, "WVS_demographics.json"), "r"))[args.country], len(generators)) # select some profiles for demographic responses
    # cultural_experts = random.sample(expert_roles, len(generators))  # use some experts as cultural experts
    cultural_expert = "cross-cultural researcher"
    foreign_experts = random.sample(country_list, 2)
    pattern = re.compile(r"Point\s*(\d+)\s*:\s*([1-5])\b", flags=re.IGNORECASE)
    question_labels = []

    prompts_base = [[] for _ in generators]  # separate prompts for different generators
    for question in question_batch:
        question_str = question["question"] if round_idx == 1 else question[f"round_{round_idx - 1}_content"]["refined_question"]
        all_cultural_points = []
        all_point_embeds = []
        for cidx in range(8):
            if "response_{}_cultural_points".format(cidx) in question[f"round_{round_idx}_content"]:
                cultural_points = question[f"round_{round_idx}_content"]["response_{}_cultural_points".format(cidx)]
                all_cultural_points.extend([x["point"] for x in cultural_points])
                all_point_embeds.extend([x["embed"] for x in cultural_points])
    
        X = np.array(all_point_embeds)
        if X.shape[0] < 2:
            continue
        hac = AgglomerativeClustering(distance_threshold=0.2, linkage='average', metric='cosine', n_clusters=None)
        hac.fit(X)
        labels = hac.labels_
        question_labels.append(labels)

        consensus_points_str = ""
        consensus_point_embeds = []
        for i in range(len(set(labels))):
            cluster_idx = np.where(labels == i)[0]  # get the indices of points in this cluster
            cluster_points = [all_cultural_points[idx] for idx in cluster_idx]  # get the cultural points in this cluster
            cluster_embeds = X[cluster_idx]
            centroid = np.mean(cluster_embeds, axis=0, keepdims=False)
            consensus_point_embeds.append(centroid.tolist())  # compute the mean embedding for the cluster
            point_str = ", ".join([x.strip() for x in cluster_points])
            consensus_points_str += f"Point {i + 1}: [{point_str}]\n"
        question[f"round_{round_idx}_content"]["consensus_point_str"] = consensus_points_str
        question[f"round_{round_idx}_content"]["consensus_point_embeds"] = consensus_point_embeds
    
        # for cidx, profile in enumerate(culture_profiles):
        for cidx, generator in enumerate(generators):
            # demographic = demographics.format(gender = profile['sex'], age = profile['age'], education = profile['education'], occupation = profile['occupation'], social_class = profile['social_class'], income_level = profile['income_level'])
            prompt_base = rep_dist_answer_general_role.format(country = args.country, question = question_str, cultural_points = consensus_points_str)
            prompts_base[cidx].append(prompt_base)

        # for cidx, expert in enumerate(cultural_experts):
        for cidx, generator in enumerate(generators):
            prompt_base = represent_point_judgment_expert.format(country = args.country, role = cultural_expert, question = question_str, cultural_points = consensus_points_str)
            prompts_base[cidx].append(prompt_base)
            
        for cidx, foreign in enumerate(foreign_experts):
            prompt_base = represent_point_judgment_foreign.format(foreign = foreign, country = args.country, question = question_str, cultural_points = consensus_points_str)
            prompts_base[0].append(prompt_base)

    ## prompts_base is a list of prompt lists, one for each generator. For generator 0, includes 3 * len(question_batch) prompts; for other generators, includes 2 * len(question_batch) prompts.

    responses = [] # [responses from generator 1, responses from generator 2, ...] where each element is a list of responses for the question batch
    # parallelize across generators (IO-bound LLM calls); ex.map preserves input order
    def _consensus_call(item):
        gen, prompts = item
        return gen.get_batch_top_p_answer(prompts, [""] * len(prompts), max_tokens=512, temperature=0.7, top_p=0.8)
    with ThreadPoolExecutor(max_workers=len(generators)) as ex:
        responses = list(ex.map(_consensus_call, list(zip(generators, prompts_base))))

    print("shape of responses:", len(responses), [len(r) for r in responses])
    
    for qidx, question in enumerate(question_batch):
        labels = question_labels[qidx]
        one_inputs = prompts_base[0][qidx * (2 + len(foreign_experts)):(qidx + 1) * (2 + len(foreign_experts))]  # inputs for the first generator, which includes all types of respondents
        one_responses = responses[0][qidx * (2 + len(foreign_experts)):(qidx + 1) * (2 + len(foreign_experts))]  # responses from the first generator, which includes all types of respondents
        for gix in range(1, len(generators)):
            one_inputs.extend(prompts_base[gix][qidx * 2:(qidx + 1) * 2])  # only include prompts for demographic and cultural experts from other generators
            one_responses.extend(responses[gix][qidx * 2:(qidx + 1) * 2])  # only include demographic and cultural expert responses from other generators

        raw_consensus_matrix = np.zeros((len(generators) * 2 + len(foreign_experts), len(set(labels))))  # score matrix for all respondents and all consensus points

        for cidx, response in enumerate(one_responses):
            input_words += len(one_inputs[cidx].split())
            output_words += len(response.split())
            for match in pattern.finditer(response):
                point_idx = int(match.group(1)) - 1
                score = int(match.group(2))
                # judgment = 1 if score >= 4 else 0
                if point_idx >= raw_consensus_matrix.shape[1]:
                    break
                raw_consensus_matrix[cidx, point_idx] = score

        likert_consensus_matrix = raw_consensus_matrix.copy()
        raw_consensus_matrix[raw_consensus_matrix < 4] = 0  # convert to binary matrix, 0 or 1
        raw_consensus_matrix[raw_consensus_matrix >= 4] = 1  #
        final_consensus = majority_consensus(raw_consensus_matrix)

        question[f"round_{round_idx}_content"]["likert_consensus"] = likert_consensus_matrix.tolist()
        question[f"round_{round_idx}_content"]["consensus_scores"] = final_consensus
    return input_words, output_words

if __name__ == "__main__":
    args = parse_arguments()
    print("Running with args: ", args)
    random.seed(2026)

    if args.stage == 1:
        init_rep_dist_questions(args)
    
    if args.stage == 2:
        rep_dist_optimization(args)
    
    if args.stage == 3:
        rep_dist_optimization_multigenerator(args)