import argparse
from eval.hf_runner import run_hf_eval


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data-related parameters
    parser.add_argument("-q", "--questions_file", type=str)
    parser.add_argument("-n", "--num_questions", type=int, default=None)
    parser.add_argument("-db", "--db_type", type=str, required=True)
    parser.add_argument("-d", "--use_private_data", action="store_true")
    # model-related parameters
    parser.add_argument("-g", "--model_type", type=str, required=True)
    parser.add_argument("-m", "--model", type=str)
    parser.add_argument("-a", "--adapter", type=str)
    parser.add_argument("--api_url", type=str)
    # inference-technique-related parameters
    parser.add_argument("-f", "--prompt_file", nargs="+", type=str, required=True)
    parser.add_argument("-b", "--num_beams", type=int, default=4)
    parser.add_argument("-c", "--num_columns", type=int, default=20)
    parser.add_argument("-s", "--shuffle_metadata", action="store_true")
    parser.add_argument("-k", "--k_shot", action="store_true")
    # execution-related parameters
    parser.add_argument("-o", "--output_file", nargs="+", type=str, required=True)
    parser.add_argument("-p", "--parallel_threads", type=int, default=5)
    parser.add_argument("-t", "--timeout_gen", type=float, default=30.0)
    parser.add_argument("-u", "--timeout_exec", type=float, default=10.0)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--upload_url", type=str)


    args = parser.parse_args(
        [
            "-db",
            "postgres",
            "-g",
            "hf",
            "-f",
            "prompts/prompt_sqlcoder.md",
            "-o",
            "output.txt",
            "--questions_file",
            "data/questions_gen_postgres.csv",
            '--model',
            'ahmedheakl/sqlcoder-7b-2-ArabicSQLV6'
        ]
    )

    args.upload_url = ""
    
    run_hf_eval(args)
        

