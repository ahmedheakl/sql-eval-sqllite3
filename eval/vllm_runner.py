import sqlparse
from vllm import LLM, SamplingParams
from eval.eval import compare_query_results
import pandas as pd
from utils.pruning import prune_metadata_str
from utils.questions import prepare_questions_df
import time
import torch
from transformers import AutoTokenizer
from tqdm import tqdm


def generate_prompt(prompt_file, question, db_name, public_data):
    with open(prompt_file, "r") as f:
        prompt = f.read()

    pruned_metadata_str = prune_metadata_str(question, db_name, public_data)
    prompt = prompt.format(
        user_question=question, table_metadata_string=pruned_metadata_str
    )
    return prompt


def run_vllm_eval(args):
    # get params from args
    questions_file = args.questions_file
    prompt_file = args.prompt_file
    num_questions = args.num_questions
    public_data = not args.use_private_data
    model_name = args.model
    output_file = args.output_file
    num_beams = args.num_beams

    # get questions
    print(
        f"Preparing {'all' if num_questions is None else num_questions} questions from {questions_file}"
    )
    df = prepare_questions_df(questions_file, num_questions)

    # create a prompt for each question
    df["prompt"] = df[["question", "db_name"]].apply(
        lambda row: generate_prompt(
            prompt_file, row["question"], row["db_name"], public_data
        ),
        axis=1,
    )

    print(f"Preparing {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = LLM(model=model_name, tensor_parallel_size=torch.cuda.device_count())

    sampling_params = SamplingParams(
        n=1,
        best_of=num_beams,
        use_beam_search=num_beams != 1,
        stop_token_ids=[tokenizer.eos_token_id],
        max_tokens=300,
        temperature=0,
    )

    print(f"Generating {len(df)} completions")
    start_time = time.time()
    outputs = llm.generate(df["prompt"].tolist(), sampling_params)
    time_taken = time.time() - start_time
    print(f"Time taken: {time_taken:.1f}s")

    df["generated_query"] = ""
    df["correct"] = 0
    df["exact_match"] = 0
    df["error_db_exec"] = 0
    df["error_msg"] = ""
    total_correct = 0
    with tqdm(total=len(df)) as pbar:
        for i, output in enumerate(outputs):
            generated_query = output.outputs[0].text.split(";")[0].strip()
            normalized_query = sqlparse.format(
                generated_query, keyword_case="upper", strip_whitespace=True
            )
            df.loc[i, "generated_query"] = normalized_query
            df.loc[i, "latency_seconds"] = time_taken / len(df)
            row = df.iloc[i]
            golden_query = row["query"]
            db_name = row["db_name"]
            question = row["question"]
            query_category = row["query_category"]
            exact_match = correct = 0
            db_creds = {
                "host": "localhost",
                "port": 5432,
                "user": "postgres",
                "password": "postgres",
                "database": db_name,
            }
            try:
                exact_match, correct = compare_query_results(
                    query_gold=golden_query,
                    query_gen=generated_query,
                    db_name=db_name,
                    db_creds=db_creds,
                    question=question,
                    query_category=query_category,
                )
                df.loc[i, "exact_match"] = int(exact_match)
                df.loc[i, "correct"] = int(correct)
                df.loc[i, "error_msg"] = ""
                if correct:
                    total_correct += 1
            except Exception as e:
                df.loc[i, "error_db_exec"] = 1
                df.loc[i, "error_msg"] = f"QUERY EXECUTION ERROR: {e}"
            pbar.update(1)
            pbar.set_description(
                f"Correct so far: {total_correct}/{(i+1)} ({100*total_correct/(i+1):.2f}%)"
            )
    del df["prompt"]
    print(df.groupby("query_category")[["exact_match", "correct"]].mean())
    df = df.sort_values(by=["db_name", "query_category", "question"])
    df.to_csv(output_file, index=False, float_format="%.2f")