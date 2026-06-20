import os
import json
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering

from models import *

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--country', type=str, default='China')
    parser.add_argument('--main_generator', type=str, default='openai/gpt-4o-mini')
    parser.add_argument('--select_score', type=str, default='both_multiply')
    parser.add_argument('--sample_num', type=int, default=1000)
    parser.add_argument('--prefix', type=str, default="represent")
    parser.add_argument('--cluster_threshold', type=float, default=0.7)
    parser.add_argument('--output_dir', type=str, default='../data')
    parser.add_argument('--round', type=int, default=1)
    return parser.parse_args()

def compute_represent_scores(labels, cluster_centers, lines, args):
    represent_score = [0 for _ in range(len(lines))]
    topic_consensus_count = defaultdict(list)
    for lidx, line in tqdm(enumerate(lines), desc="Computing represent scores"):
        if "consensus_scores" not in line:
            return represent_score
        consensus_scores = [v for k, v in line["consensus_scores"].items()]
        consensus_count = sum([1 if x > 0.9 else 0 for x in consensus_scores]) # 1 for agree, 0 for disagree
        consensus_scores = [1 if x > 0.9 else -1 for x in consensus_scores] # 1 for agree, -1 for disagree  -1 as a penalty for the represent score
        represent_score[lidx] = sum(consensus_scores)
        topic_consensus_count[line["topic"]].append(consensus_count)
    
    normalized_represent_score = [0 for _ in range(len(lines))]
    for lidx, line in tqdm(enumerate(lines), desc="Computing normalized represent scores"):
        norm = np.mean(topic_consensus_count[line["topic"]])
        normalized_represent_score[lidx] = represent_score[lidx] / (norm + 1e-9)

    rep_center_score = [0 for _ in range(len(lines))]
    for label, center in cluster_centers.items():
        rep_center_score[center] = normalized_represent_score[center]
    return rep_center_score

def compute_distinctive_scores(labels, cluster_centers, lines, args):
    distinctive_score = [0 for _ in range(len(lines))]
    for label, center in tqdm(cluster_centers.items(), desc="Computing distinctive scores"):
        line = lines[center]
        one_distinctive_score = [v for k, v in line["distinctiveness_scores"].items()]
        distinctive_score[center] = np.mean(one_distinctive_score)
        # main_country_answer_embed = line["answer_embed"]

        # if "other_country_responses" in line:
        #     other_country_answer_embeds = []
        #     for response in line["other_country_responses"]:
        #         other_country_answer_embeds.append(response["response_embed"])
        # else:
        #     return distinctive_score

        # cosine_distance = [1 - np.dot(main_country_answer_embed, other_country_answer_embed) / (np.linalg.norm(main_country_answer_embed) * np.linalg.norm(other_country_answer_embed)) for other_country_answer_embed in other_country_answer_embeds]
        # distinctive_score[center] = sum(cosine_distance) / (len(cosine_distance) + 1e-6)
    return distinctive_score

def select_samples(represent_score, distinctive_score, lines, sample_num=1000):
    if args.select_score == "rep_dist":
        final_score = [represent_score[i] * distinctive_score[i] for i in range(len(lines))]
    elif args.select_score == "only_rep":
        final_score = represent_score
    elif args.select_score == "only_dist":
        final_score = distinctive_score

    ## display the distribution of the scores
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.hist(represent_score, bins=100, color='blue', alpha=0.7, edgecolor='black')
    plt.title("Distribution of Represent Score")
    
    plt.subplot(1, 3, 2)
    plt.hist(distinctive_score, bins=100, color='blue', alpha=0.7, edgecolor='black')
    plt.title("Distribution of Distinctive Score")

    plt.subplot(1, 3, 3)
    plt.hist(final_score, bins=100, color='blue', alpha=0.7, edgecolor='black')
    plt.title("Distribution of Final Score")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, args.country, "figures", f"{args.prefix}_{args.main_generator.split('/')[-1]}_cluster_threshold_{args.cluster_threshold}_round_{args.round}.png"))
    plt.close()
        
    sorted_idx = np.argsort(final_score)[::-1]
    fw = open(os.path.join(args.output_dir, args.country, f"{args.prefix}_{args.main_generator.split('/')[-1]}_cluster_threshold_{args.cluster_threshold}_round_{args.round}/{args.select_score}_{sample_num}.jsonl"), 'w')
    for idx in sorted_idx[:sample_num]:
        line = lines[idx]
        if "answer_embed" in line:
            del line["answer_embed"]
        if "other_country_responses" in line:
            for response in line["other_country_responses"]:
                if "response_embed" in response:
                    del response["response_embed"]
        fw.write(json.dumps(line) + '\n')
    fw.close()

if __name__ == "__main__":
    args = parse_arguments()
    print("Running with args: ", args)

    ## step 0. prepare the dataset, compute the text embedding
    if not os.path.exists(f"{args.output_dir}/{args.country}/embeds/{args.prefix}_answers_{args.main_generator.split('/')[-1]}_round_{args.round}_embed.jsonl"):
        print("Computing text embeddings...")
        os.makedirs(f"{args.output_dir}/{args.country}/embeds", exist_ok=True)
        lines = open(f"{args.output_dir}/{args.country}/{args.prefix}_answers_{args.main_generator.split('/')[-1]}.jsonl", 'r').readlines()
        lines = [json.loads(line) for line in lines]

        fw = open(f"{args.output_dir}/{args.country}/embeds/{args.prefix}_answers_{args.main_generator.split('/')[-1]}_round_{args.round}_embed.jsonl", 'w')
        client = create_aoai_client()
        
        def embed_line(line):
            round_answer = line[f"round_{args.round}_content"]
            if "rep" in args.prefix:
                text = round_answer["integrated_answer"]
            else:
                text = round_answer["main_country_response"]

            text_embed = client.embeddings.create(input=[text], model="text-embedding-3-small").data[0].embedding
            real_question = line["question"] if args.round == 1 else line[f"round_{args.round-1}_content"]["refined_question"]
            new_line = {"question": line['question'], "real_question": real_question, "answer": text, "answer_embed": text_embed, "round_idx": args.round, "topic": line["topic"]}
            if "consensus_scores" in round_answer:
                new_line["consensus_scores"] = round_answer["consensus_scores"]
            if "distinctiveness_scores" in round_answer:
                new_line["distinctiveness_scores"] = round_answer["distinctiveness_scores"]

            if "other_country_responses" in round_answer:
                for response in round_answer["other_country_responses"]:
                    response_embed = client.embeddings.create(input=[response["response"]], model="text-embedding-3-small").data[0].embedding
                    response["response_embed"] = response_embed
                new_line["other_country_responses"] = round_answer["other_country_responses"]
            return new_line
        
        for start_idx in tqdm(range(0, len(lines), 32), desc="Computing text embeddings"):
            batch_lines = lines[start_idx:start_idx + 32]
            batch_new_lines = [None] * len(batch_lines)
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = {executor.submit(embed_line, batch_lines[idx]): idx for idx in range(len(batch_lines))}

                for future in as_completed(futures):
                    idx = futures[future]
                    batch_new_lines[idx] = future.result()
            
            for idx, new_line in enumerate(batch_new_lines):
                fw.write(json.dumps(new_line) + '\n')    
        fw.close()
        exit()
    else:
        print("Loading precomputed text embeddings...")
        lines = open(f"{args.output_dir}/{args.country}/embeds/{args.prefix}_answers_{args.main_generator.split('/')[-1]}_round_{args.round}_embed.jsonl", 'r').readlines()
        lines = [json.loads(line) for line in lines]
        content_lines = open(f"{args.output_dir}/{args.country}/{args.prefix}_answers_{args.main_generator.split('/')[-1]}.jsonl", 'r').readlines()
        content_lines = [json.loads(line) for line in content_lines]
        print("#new lines: ", len(lines), ", #raw lines: ", len(content_lines))
        for idx in tqdm(range(min(len(lines), len(content_lines))), desc="check the consistency among new lines and original"):
            assert lines[idx]['question'] == content_lines[idx]['question']
    
    ## step 1. cluster the samples to deduplicate and ensure the diversity
    print("Clustering samples...")
    X = np.array([line['answer_embed'] for line in lines])
    hac = AgglomerativeClustering(distance_threshold=args.cluster_threshold, linkage='average', metric='cosine', n_clusters=None)
    hac.fit(X)
    labels = hac.labels_
    print("Total number: ", len(lines), "Cluster number: ", len(set(labels)))
    cluster_centers = {}
    for label in set(labels):
        idxs = [i for i, l in enumerate(labels) if l == label]
        embeds = [X[i] for i in idxs]
        cosine_sim = np.dot(embeds, np.array(embeds).T) / (np.linalg.norm(embeds, axis=1) * np.linalg.norm(embeds, axis=1)[:, np.newaxis]) # (n, n)
        mean_cosine_sim = np.mean(cosine_sim, axis=1)
        center = idxs[np.argmax(mean_cosine_sim)]
        cluster_centers[label] = center

    ## step 2. compute the represent score
    print("Computing represent scores...")
    represent_score = compute_represent_scores(labels, cluster_centers, lines, args)

    ## step 3. compute the distinctive score
    print("Computing distinctive scores...")
    distinctive_score = compute_distinctive_scores(labels, cluster_centers, lines, args)

    ## step 4. compute the final score and select appropriate samples
    print("Computing final scores and selecting samples...")
    os.makedirs(os.path.join(args.output_dir, args.country, f"{args.prefix}_{args.main_generator.split('/')[-1]}_cluster_threshold_{args.cluster_threshold}_round_{args.round}"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, args.country, "figures"), exist_ok=True)
    select_samples(represent_score, distinctive_score, lines, args.sample_num)