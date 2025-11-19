import csv
import random
from dotenv import load_dotenv
from typing import List

from prompt_optimizer_v3 import (
    OpenAILLM,
    generate_seed_prompt,
    one_hop_improve,
    beam_search,
    random_walk_search,
    evaluate_metric_string_match,
    evaluate_metric_critic_lm,
)


from moves import (
    VerboseMove,
    ChainOfThoughtMove,
    ShortenMove,
    ReorderMove,
    AddExamplesMove,
    AddConstraintsMove,
    RephraseInputOutputInstructionsMove,
    AddDefinitionsMove,
    RoleAssignmentMove
)


from datasets import (
    generate_sentiment_dataset,
    generate_qa_dataset,
    generate_summarization_dataset,
    generate_reasoning_dataset,
    generate_nli_dataset,
)


load_dotenv()

#for reproducibility
SEED = 42
random.seed(SEED)

def run_rq1_experiments(
    task_name: str,
    seed_prompt: str,
    train_set: List[dict],
    dev_set: List[dict],
    test_set: List[dict],
    llm: OpenAILLM,
    moves,
    task_type: str,
    evaluation_metric,
    random_walk_steps=5,
    beam_width=2,
    beam_depth=2,
):
    print(f"\n==================== RQ1: {task_name} ====================\n")

    model = llm.get_model()
    results = {}

    # ————————————————————————————————
    # SEED BASELINE
    # ————————————————————————————————
    seed_score = evaluation_metric(seed_prompt, dev_set, model, task_type)
    seed_test_score = evaluation_metric(seed_prompt, test_set, model, task_type)

    #seed evaluation first...

    results["seed"] = {
        "prompt": seed_prompt,
        "dev_score": seed_score,
        "test_score": seed_test_score,
        "path": []
    }

    # ————————————————————————————————
    # ONE-HOP IMPROVEMENT
    # ————————————————————————————————

    #one hop imporivement..!

    improved = one_hop_improve(model, seed_prompt)
    improved_dev = evaluation_metric(improved, dev_set, model, task_type)
    improved_test = evaluation_metric(improved, test_set, model, task_type)

    results["one_hop"] = {
        "prompt": improved,
        "dev_score": improved_dev,
        "test_score": improved_test,
        "path": []
    }


    rw_best = random_walk_search(
        seed_prompt, train_set, dev_set, llm, moves,
        steps=random_walk_steps,
        task_type=task_type,
        evaluation_metric=evaluation_metric
    )

    rw_dev_score = rw_best.score
    rw_test_score = evaluation_metric(rw_best.prompt_text, test_set, model, task_type)

    #again store dev set and the test set both...
    results["random_walk"] = {
        "prompt": rw_best.prompt_text,
        "dev_score": rw_dev_score,
        "test_score": rw_test_score,
        "path": rw_best.get_path(),
    }


    #now we use the actual beam search optimizer!!!
    beam_best = beam_search(
        seed_prompt, train_set, dev_set, llm, moves,
        beam_width=beam_width,
        depth=beam_depth,
        task_type=task_type,
        evaluation_metric=evaluation_metric
    )

    beam_dev_score = beam_best.score
    beam_test_score = evaluation_metric(beam_best.prompt_text, test_set, model, task_type)

    results["beam_search"] = {
        "prompt": beam_best.prompt_text,
        "dev_score": beam_dev_score,
        "test_score": beam_test_score,
        "path": beam_best.get_path(),
    }

    return results




def run_rq1_all_tasks_and_export():
    print("\n=============== RUNNING RQ1 ON ALL DATASETS ===============\n")

    TASKS = [
        ("sentiment", generate_sentiment_dataset, "classification", False),
        ("qa", generate_qa_dataset, "qa", False),
        ("summarization", generate_summarization_dataset, "summarization", True),
        ("reasoning", generate_reasoning_dataset, "reasoning", True),
        ("nli", generate_nli_dataset, "nli", True),
    ]


    llm = OpenAILLM()
    model = llm.get_model()


    moves = [
        VerboseMove(model),
        ShortenMove(model),
        ReorderMove(model),
        AddExamplesMove(model),
    ]

    csv_filename = "rq1_results.csv"

    with open(csv_filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["task", "method", "dev_score", "test_score", "prompt", "path"],
            quoting=csv.QUOTE_ALL,
            escapechar="\\",
            doublequote=True,
        )
        writer.writeheader()

    for task_name, dataset_fn, task_type, use_critic in TASKS:
        print(f"\n============ Loading dataset for {task_name} ============\n")

        full_set = dataset_fn()
        random.shuffle(full_set)

        n = len(full_set)
        train_end = max(1, int(0.25 * n))
        dev_end = train_end + max(1, int(0.25 * n))

        train_set = full_set[:train_end]
        dev_set = full_set[train_end:dev_end]
        test_set = full_set[dev_end:]

        # Generate seed prompt using all train examples
        seed_prompt = generate_seed_prompt(train_set, task_type, model)

        evaluation_metric = (
            evaluate_metric_critic_lm if use_critic else evaluate_metric_string_match
        )


        results = run_rq1_experiments(
            task_name=task_name,
            seed_prompt=seed_prompt,
            train_set=train_set,
            dev_set=dev_set,
            test_set=test_set,
            llm=llm,
            moves=moves,
            task_type=task_type,
            evaluation_metric=evaluation_metric
        )

        with open(csv_filename, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["task", "method", "dev_score", "test_score", "prompt", "path"],
                quoting=csv.QUOTE_ALL,
                escapechar="\\",
                doublequote=True,
            )

            for method_name, result in results.items():
                writer.writerow({
                    "task": task_name,
                    "method": method_name,
                    "dev_score": result["dev_score"],
                    "test_score": result["test_score"],
                    "prompt": result["prompt"],
                    "path": " -> ".join(result["path"]) if result["path"] else ""
                })

        print(f"\nTask {task_name} finished — results appended to CSV.\n")

    print(f"\n🎉 RQ1 COMPLETE — All results saved to {csv_filename}\n")



def run_specific_rq1_all_task_and_export(task_name: str, output_file: str = None):


    print(f"\n=============== RUNNING RQ1 FOR TASK: {task_name} ===============\n")

    # Registered tasks
    TASK_MAP = {
        "sentiment":      (generate_sentiment_dataset, "classification", False),
        "qa":             (generate_qa_dataset, "qa", False),
        "summarization":  (generate_summarization_dataset, "summarization", True),
        "reasoning":      (generate_reasoning_dataset, "reasoning", True),
        "nli":            (generate_nli_dataset, "nli", True),
    }

    if task_name not in TASK_MAP:
        raise ValueError(f"Invalid task: {task_name}. Must be one of: {list(TASK_MAP.keys())}")

    dataset_fn, task_type, use_critic = TASK_MAP[task_name]


    if output_file is None:
        output_file = f"rq1_results_{task_name}.csv"


    llm = OpenAILLM()
    model = llm.get_model()


    moves = [
        VerboseMove(model),
        ShortenMove(model),
        ReorderMove(model),
        AddExamplesMove(model),
    ]


    full_set = dataset_fn()
    random.shuffle(full_set)

    n = len(full_set)
    train_end = max(1, int(0.25 * n))
    dev_end = train_end + max(1, int(0.25 * n))

    train_set = full_set[:train_end]
    dev_set   = full_set[train_end:dev_end]
    test_set  = full_set[dev_end:]

    print(f"Dataset size = {n}  (train={len(train_set)}, dev={len(dev_set)}, test={len(test_set)})")


    seed_prompt = generate_seed_prompt(train_set, task_type, model)


    evaluation_metric = (
        evaluate_metric_critic_lm if use_critic else evaluate_metric_string_match
    )


    results = run_rq1_experiments(
        task_name=task_name,
        seed_prompt=seed_prompt,
        train_set=train_set,
        dev_set=dev_set,
        test_set=test_set,
        llm=llm,
        moves=moves,
        task_type=task_type,
        evaluation_metric=evaluation_metric
    )


    rows = []
    for method_name, result in results.items():
        rows.append({
            "task": task_name,
            "method": method_name,
            "dev_score": result["dev_score"],
            "test_score": result["test_score"],
            "prompt": result["prompt"],
            "path": " -> ".join(result["path"]) if result["path"] else ""
        })

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["task", "method", "dev_score", "test_score", "prompt", "path"],
            quoting=csv.QUOTE_ALL,
            escapechar="\\",
            doublequote=True,
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n🎉 FINISHED TASK {task_name}. Results saved to {output_file}\n")




if __name__ == "__main__":
    import sys

    # Usage:
    # python rq1.py sentiment results_sentiment.csv
    # python rq1.py qa
    # python rq1.py

    args = sys.argv[1:]

    if len(args) == 0:
        print("Usage: python rq1.py <task_name> [output_csv]")
        print("Tasks: sentiment, qa, summarization, reasoning, nli")
        sys.exit(1)

    task = args[0].strip().lower()
    output_csv = args[1] if len(args) > 1 else None

    run_specific_rq1_all_task_and_export(task, output_csv)
