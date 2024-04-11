from sqlite3 import connect
import numpy as np
from typing import Dict, List, Optional
import json
import os
import pandas as pd
import re
from utils.questions import prepare_questions_df
import itertools
from sqlalchemy import create_engine

db_path = "data/defog_data/{db_name}/{db_name}.db"


def to_prompt_schema(
    md: Dict[str, List[Dict[str, str]]], seed: Optional[int] = None
) -> str:
    """
    Return a DDL statement for creating tables from a metadata dictionary
    `md` has the following structure:
        {'table1': [
            {'column_name': 'col1', 'data_type': 'int', 'column_description': 'primary key'},
            {'column_name': 'col2', 'data_type': 'text', 'column_description': 'not null'},
            {'column_name': 'col3', 'data_type': 'text', 'column_description': ''},
        ],
        'table2': [
        ...
        ]},
    This is just for converting the dictionary structure of one's metadata into a string
    for pasting into prompts, and not meant to be used to initialize a database.
    seed is used to shuffle the order of the tables when not None
    """
    md_create = ""
    table_names = list(md.keys())
    if seed:
        np.random.seed(seed)
        np.random.shuffle(table_names)
    for table in table_names:
        md_create += f"CREATE TABLE {table} (\n"
        columns = md[table]
        if seed:
            np.random.seed(seed)
            np.random.shuffle(columns)
        for i, column in enumerate(columns):
            col_name = column["column_name"]
            # if column name has spaces, wrap it in double quotes
            if " " in col_name:
                col_name = f'"{col_name}"'
            dtype = column["data_type"]
            col_desc = column.get("column_description", "").replace("\n", " ")
            if col_desc:
                col_desc = f" --{col_desc}"
            if i < len(columns) - 1:
                md_create += f"  {col_name} {dtype},{col_desc}\n"
            else:
                # avoid the trailing comma for the last line
                md_create += f"  {col_name} {dtype}{col_desc}\n"
        md_create += ");\n"
    return md_create

def get_db(db_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, f"data/defog_data/{db_name}/{db_name}.json")
    with open(file_path, "r") as f:
        db_schema = json.load(f)
    return db_schema

def convert_postgres_to_sqlite3(sql):
    """
    Converts a Postgres SQL query to a compatible SQLite3 query.

    Args:
        sql: The Postgres SQL query string.

    Returns:
        The converted SQLite3 query string.
    """
    # Replace unsupported features
    sql = sql.replace("::float", "/1.0")  # Cast to float using division
    sql = re.sub(r"(?i)ILIKE", "LIKE", sql, flags=re.IGNORECASE)
    sql = re.sub(r"(?i)extract\(year from ", "strftime('%Y', ", sql, flags=re.IGNORECASE)
    sql = re.sub(r"(?i)date_trunc\('month',", "strftime('%Y-%m',", sql, flags=re.IGNORECASE)
    sql = re.sub(r"(?i)date_trunc\('year',", "strftime('%Y',", sql, flags=re.IGNORECASE)
    sql = re.sub(r"(?i)date_trunc\('week',", "strftime('%Y-%W',", sql, flags=re.IGNORECASE)
    sql = re.sub(r"(?i)date_trunc\('day',", "date(", sql, flags=re.IGNORECASE)
    sql = re.sub(r"(?i)date_trunc\('hour',", "strftime('%Y-%m-%d %H',", sql, flags=re.IGNORECASE)
    sql = re.sub(r"(?i)date_trunc\('minute',", "strftime('%Y-%m-%d %H:%M',", sql, flags=re.IGNORECASE)
    sql = re.sub(r"(?i)date_trunc\('second',", "strftime('%Y-%m-%d %H:%M:%S',", sql, flags=re.IGNORECASE)
    sql = sql.replace("interval '", "INTERVAL '")  # INTERVAL format
    sql = re.sub(r"(?i)to_timestamp\(", "datetime(", sql, flags=re.IGNORECASE)
    sql = re.sub(r"(?i)extract\(MONTH FROM ", "strftime('%m', ", sql, flags=re.IGNORECASE)
    sql = re.sub(r"(?i)extract\(YEAR FROM ", "strftime('%Y', ", sql, flags=re.IGNORECASE)
    sql = re.sub(r"(?i)to_date\(", "date(", sql, flags=re.IGNORECASE)
    interval = re.search(r"interval '(.+?)'", sql, flags=re.IGNORECASE)
    if interval:
        interval = interval.group(1)
        sql = sql.replace(f"INTERVAL '{interval}'", f"strftime('%s', 'now') - strftime('%s', 'now', '-{interval}')")
    sql = sql.replace("::date", "")  # Remove type casting to date
    sql = sql.replace("::TIME", "")  # Remove type casting to time
    # convert date_part to strftime
    date_part = re.search(r"date_part\('(.+?)', (.+?)\)", sql, flags=re.IGNORECASE)
    if date_part:
        part = date_part.group(1)
        date = date_part.group(2)
        sql = sql.replace(date_part.group(0), f"strftime('%{part}', {date})")
    sql = re.sub(r"(FROM|JOIN) (\w+)\.", r"\1 ", sql, flags=re.IGNORECASE)
    # replace to_char with strftime
    to_char = re.search(r"to_char\((.+?), '(.+?)'\)", sql, flags=re.IGNORECASE)
    if to_char:
        date = to_char.group(1)
        fmt = to_char.group(2)
        sql = sql.replace(to_char.group(0), f"strftime('{fmt}', {date})")

    # replace NOW() with datetime('now')
    sql = re.sub(r"(?i)NOW\(\)", "datetime('now')", sql, flags=re.IGNORECASE)
    
    return sql

def find_bracket_indices(s: str, start_index: int = 0) -> "tuple[int, int]":
    start = s.find("{", start_index)
    end = s.find("}", start + 1)
    if start == -1 or end == -1:
        return (-1, -1)
    return (start, end)

def get_all_minimal_queries(query: str) -> "list[str]":
    """
    extrapolate all possible queries
    - split by semicolon. this is to accommodate queries where joins to other tables are also acceptable.
    - expand all column permutations if there are braces { } in it. eg:
    ```sql
        SELECT {user.id, user.name} FROM user;
    ```
    Would be expanded to:
    ```sql
        SELECT user.id FROM user;
        SELECT user.name FROM user;
        SELECT user.id, user.name FROM user;
    ```
    """
    queries = query.split(";")
    result_queries = []
    for query in queries:
        query = query.strip()
        if query == "":
            continue
        start, end = find_bracket_indices(query, 0)
        if (start, end) == (-1, -1):
            result_queries.append(query)
            continue
        else:
            # get all possible column subsets
            column_options = query[start + 1 : end].split(",")
            column_combinations = list(
                itertools.chain.from_iterable(
                    itertools.combinations(column_options, r)
                    for r in range(1, len(column_options) + 1)
                )
            )
            for column_tuple in column_combinations:
                left = query[:start]
                column_str = ", ".join(column_tuple)
                right = query[end + 1 :]
                # change group by size dynamically if necessary
                if right.find("GROUP BY {}"):
                    right = right.replace("GROUP BY {}", f"GROUP BY {column_str}")
                result_queries.append(left + column_str + right)
    return result_queries

def normalize_table(
    df: pd.DataFrame, query_category: str, question: str, sql: str = None
) -> pd.DataFrame:
    """
    Normalizes a dataframe by:
    1. removing all duplicate rows
    2. sorting columns in alphabetical order
    3. sorting rows using values from first column to last (if query_category is not 'order_by' and question does not ask for ordering)
    4. resetting index
    """
    # remove duplicate rows, if any
    df = df.drop_duplicates()

    # sort columns in alphabetical order of column names
    sorted_df = df.reindex(sorted(df.columns), axis=1)

    # check if query_category is 'order_by' and if question asks for ordering
    has_order_by = False
    pattern = re.compile(r"\b(order|sort|arrange)\b", re.IGNORECASE)
    in_question = re.search(pattern, question.lower())  # true if contains
    if query_category == "order_by" or in_question:
        has_order_by = True

        if sql:
            # determine which columns are in the ORDER BY clause of the sql generated, using regex
            pattern = re.compile(r"ORDER BY[\s\S]*", re.IGNORECASE)
            order_by_clause = re.search(pattern, sql)
            if order_by_clause:
                order_by_clause = order_by_clause.group(0)
                # get all columns in the ORDER BY clause, by looking at the text between ORDER BY and the next semicolon, comma, or parantheses
                pattern = re.compile(r"(?<=ORDER BY)(.*?)(?=;|,|\)|$)", re.IGNORECASE)
                order_by_columns = re.findall(pattern, order_by_clause)
                order_by_columns = (
                    order_by_columns[0].split() if order_by_columns else []
                )
                order_by_columns = [
                    col.strip().rsplit(".", 1)[-1] for col in order_by_columns
                ]

                ascending = False
                # if there is a DESC or ASC in the ORDER BY clause, set the ascending to that
                if "DESC" in [i.upper() for i in order_by_columns]:
                    ascending = False
                elif "ASC" in [i.upper() for i in order_by_columns]:
                    ascending = True

                # remove whitespace, commas, and parantheses
                order_by_columns = [col.strip() for col in order_by_columns]
                order_by_columns = [
                    col.replace(",", "").replace("(", "") for col in order_by_columns
                ]
                order_by_columns = [
                    i
                    for i in order_by_columns
                    if i.lower()
                    not in ["desc", "asc", "nulls", "last", "first", "limit"]
                ]

                # get all columns in sorted_df that are not in order_by_columns
                other_columns = [
                    i for i in sorted_df.columns.tolist() if i not in order_by_columns
                ]

                # only choose order_by_columns that are in sorted_df
                order_by_columns = [
                    i for i in order_by_columns if i in sorted_df.columns.tolist()
                ]
                sorted_df = sorted_df.sort_values(
                    by=order_by_columns + other_columns, ascending=ascending
                )

    if not has_order_by:
        # sort rows using values from first column to last
        sorted_df = sorted_df.sort_values(by=list(sorted_df.columns))

    # reset index
    sorted_df = sorted_df.reset_index(drop=True)
    return sorted_df

def compare_df(
    df_gold: pd.DataFrame,
    df_gen: pd.DataFrame,
    query_category: str,
    question: str,
    query_gold: str = None,
    query_gen: str = None,
    tolerance: float = 1e-5,
) -> bool:
    """
    Compares two dataframes and returns True if they are the same, else False.
    query_gold and query_gen are the original queries that generated the respective dataframes.
    """
    # drop duplicates to ensure equivalence
    try:
        if df_gold.shape == df_gen.shape and (df_gold.values == df_gen.values).all():
            return True
    except:
        if df_gold.shape == df_gen.shape and (df_gold.values == df_gen.values):
            return True

    df_gold = normalize_table(df_gold, query_category, question, query_gold)
    df_gen = normalize_table(df_gen, query_category, question, query_gen)
    
    if df_gold.shape != df_gen.shape:
        return False
    
    for col in df_gold.columns:
        if df_gold[col].dtype == np.float64:
            if not np.allclose(df_gold[col], df_gen[col], atol=tolerance):
                return False
        else:
            if not df_gold[col].equals(df_gen[col]):
                return False
            
    return True



def main():
    eval_paths = ["data/instruct_basic_postgres.csv", "data/instruct_advanced_postgres.csv", "data/questions_gen_postgres.csv"]
    # eval_paths = ["data/instruct_basic_postgres.csv", ]
    for eval_path in eval_paths:
        eval_df = pd.read_csv(eval_path)
        worked = 0
        for i, row in eval_df.iterrows():
            
            db_name = row["db_name"]
            conn = connect(db_path.format(db_name=db_name))
            c = conn.cursor()
            old_query = row["query"]
            minimals = get_all_minimal_queries(old_query)
            engine = create_engine(f'postgresql://postgres:postgres@localhost/{db_name}')

            local_worked = 0
            for minimal in minimals:
                try:
                    gold_df = pd.read_sql_query(minimal, engine)
                    query = convert_postgres_to_sqlite3(minimal)
                    gen_df = pd.read_sql_query(query, conn)
                    correct = compare_df(gold_df, gen_df, row["query_category"], row["question"])
                    # if not correct:
                    #     print(f"Error in {db_name} row {i}\nQuery: {query}\nOld Query: {old_query}\n")
                    local_worked += correct
                except Exception as e:
                    pass
                    # print(f"Error in {db_name}: {e}\nQuery: {query}\nOld Query: {old_query}\n")
            
            worked += (local_worked / len(minimals))


        print(eval_path)
        print(f"Worked: {worked}/{len(eval_df)}. Percentage: {worked * 100/len(eval_df):.2f}%\n\n")



       


if __name__ == "__main__":
    main()