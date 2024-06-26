from typing import Optional

from eval.eval import compare_query_results
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)
from utils.gen_prompt import generate_prompt
from utils.questions import prepare_questions_df
from utils.creds import db_creds_all
from tqdm import tqdm
from psycopg2.extensions import QueryCanceledError
from time import time
import gc

def get_tokenizer_model(model_name: Optional[str], adapter_path: Optional[str]):
    """
    Load a HuggingFace tokenizer and model.
    You may supply either a normal huggingface model name, or a peft adapter path.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
    )
    return tokenizer, model


def run_hf_eval(args):
    # get params from args
    questions_file = args.questions_file
    prompt_file_list = args.prompt_file
    num_questions = args.num_questions
    public_data = not args.use_private_data
    model_name = args.model
    adapter_path = args.adapter
    output_file_list = args.output_file
    k_shot = args.k_shot
    db_type = args.db_type

    print("Preparing questions...")
    df = prepare_questions_df(questions_file, db_type, num_questions, k_shot)
    
    if model_name is None and adapter_path is None:
        raise ValueError(
            "You must supply either a model name or an adapter path to run an evaluation."
        )

    print(f"Questions prepared\nNow loading model...")
    tokenizer, model = get_tokenizer_model(model_name, adapter_path)
    model.tie_weights()

    print("model loaded\nnow generating and evaluating predictions...")

    eos_token_id = tokenizer.convert_tokens_to_ids(["```"])[0]
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    support_beam_search = True


    for prompt_file, output_file in zip(prompt_file_list, output_file_list):
        df["prompt"] = df[
            [   
                "arabic",
                "question",
                "db_name",
                "instructions",
                "k_shot_prompt",
                "glossary",
                "table_metadata_string",
                "prev_invalid_sql",
                "prev_error_msg",
                "question_0",
                "query_0",
                "question_1",
                "query_1",
            ]
        ].apply(
            lambda row: generate_prompt(
                prompt_file,
                row[args.target_column],
                row["db_name"],
                row["instructions"],
                row["k_shot_prompt"],
                row["glossary"],
                row["table_metadata_string"],
                row["prev_invalid_sql"],
                row["prev_error_msg"],
                row["question_0"],
                row["query_0"],
                row["question_1"],
                row["query_1"],
                public_data,
                args.num_columns,
                args.shuffle_metadata,
            ),
            axis=1,
        )

        total_tried = 0
        total_correct = 0
        output_rows = []

        with tqdm(total=len(df)) as pbar:
            for row in df.to_dict("records"):
                total_tried += 1
                start_time = time()

                num_beams = 1 + int(support_beam_search)

                # we set return_full_text to False so that we don't get the prompt text in the generated text
                # this simplifies our postprocessing to deal with just the truncation of the end of the query
                generated_query = pipe(
                    row["prompt"],
                    max_new_tokens=100,
                    do_sample=False,
                    num_beams=num_beams,
                    num_return_sequences=1,
                    return_full_text=False,
                    # eos_token_id=tokenizer.eos_token_id,
                    # pad_token_id=tokenizer.eos_token_id,
                )[0]["generated_text"]
                if "[SQL]" not in row["prompt"]:
                    generated_query = (
                        generated_query.split("```")[0].split(";")[0].strip() + ";"
                    )
                else:
                    generated_query = generated_query.split("[SQL]\n")[-1].split(["[/SQL]"])[0]

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                end_time = time()

                row["generated_query"] = generated_query
                row["latency_seconds"] = end_time - start_time
                golden_query = row["query"]
                db_name = row["db_name"]
                db_type = row["db_type"]
                question = row["question"]
                query_category = row["query_category"]
                table_metadata_string = row["table_metadata_string"]
                exact_match = correct = 0
                db_creds = db_creds_all[db_type]

                try:
                    exact_match, correct = compare_query_results(
                        query_gold=golden_query,
                        query_gen=generated_query,
                        db_name=db_name,
                        db_type=db_type,
                        db_creds=db_creds,
                        question=question,
                        query_category=query_category,
                        table_metadata_string=table_metadata_string,
                    )
                    row["exact_match"] = int(exact_match)
                    row["correct"] = int(correct)
                    row["error_msg"] = ""
                    if correct:
                        total_correct += 1
                except QueryCanceledError as e:
                    row["timeout"] = 1
                    row["error_msg"] = f"QUERY EXECUTION TIMEOUT: {e}"
                except Exception as e:
                    row["error_db_exec"] = 1
                    row["error_msg"] = f"QUERY EXECUTION ERROR: {e}"

                output_rows.append(row)
                pbar.update(1)
                pbar.set_description(
                    f"Correct so far: {total_correct}/{total_tried} ({100*total_correct/total_tried:.2f}%)"
                )

        output_df = pd.DataFrame(output_rows)
        del output_df["prompt"]
        print(output_df.groupby("query_category")[["correct", "error_db_exec"]].mean())
        output_df = output_df.sort_values(by=["db_name", "query_category", "question"])
        output_df.to_csv(output_file, index=False, float_format="%.2f")

