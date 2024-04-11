# this file contains all of the helper functions used for evaluations
import sqlite3
import itertools
import re
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
import numpy as np

LIKE_PATTERN = r"LIKE[\s\S]*'"

def convert_postgres_to_sqlite3(sql):
    """Converts a Postgres SQL query to a compatible SQLite3 query.
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


# for escaping percent signs in regex matches
def escape_percent(match):
    # Extract the matched group
    group = match.group(0)
    # Replace '%' with '%%' within the matched group
    escaped_group = group.replace("%", "%%")
    # Return the escaped group
    return escaped_group


# find start and end index of { } in a string. return (start, end) if found, else return (-1, -1)
def find_bracket_indices(s: str, start_index: int = 0) -> "tuple[int, int]":
    start = s.find("{", start_index)
    end = s.find("}", start + 1)
    if start == -1 or end == -1:
        return (-1, -1)
    return (start, end)


# extrapolate all possible queries from a query with { } in it
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


def clean_metadata_string(md_str: str) -> str:
    # for every line, remove all text after "--"
    md_str = "\n".join([line.split("--")[0] for line in md_str.split("\n")])
    # remove all ", \n);"
    md_str = md_str.replace(", \n);", "\n);").replace(",\n);", "\n);").strip()
    md_str = md_str.split("Here is a list of joinable columns:")[0].strip()
    return md_str


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

def query_sqllite3_db(query: str, db_name: str, timeout: float = 10.0) -> pd.DataFrame:
    """
    Queries a sqllite3 database and returns the result as a pandas dataframe.
    """
    try:
        conn = sqlite3.connect(f"data/defog_data/{db_name}/{db_name}.db")
        df = pd.read_sql_query(query, conn)
        conn.close()    
    except Exception as e:
        from psycopg2.extensions import QueryCanceledError
        raise QueryCanceledError()
        return None
    return df


def subset_df(
    df_sub: pd.DataFrame,
    df_super: pd.DataFrame,
    query_category: str,
    question: str,
    query_super: str = None,
    query_sub: str = None,
    verbose: bool = False,
) -> bool:
    """
    Checks if df_sub is a subset of df_super
    """
    if df_sub.empty:
        return False  # handle cases for empty dataframes

    # make a copy of df_super so we don't modify the original while keeping track of matches
    df_super_temp = df_super.copy(deep=True)
    matched_columns = []
    for col_sub_name in df_sub.columns:
        col_match = False
        for col_super_name in df_super_temp.columns:
            col_sub = df_sub[col_sub_name].sort_values().reset_index(drop=True)
            col_super = (
                df_super_temp[col_super_name].sort_values().reset_index(drop=True)
            )
            try:
                assert_series_equal(
                    col_sub, col_super, check_dtype=False, check_names=False
                )
                col_match = True
                matched_columns.append(col_super_name)
                # remove col_super_name to prevent us from matching it again
                df_super_temp = df_super_temp.drop(columns=[col_super_name])
                break
            except AssertionError:
                continue
        if col_match == False:
            if verbose:
                print(f"no match for {col_sub_name}")
            return False
    df_sub_normalized = normalize_table(df_sub, query_category, question, query_sub)

    # get matched columns from df_super, and rename them with columns from df_sub, then normalize
    df_super_matched = df_super[matched_columns].rename(
        columns=dict(zip(matched_columns, df_sub.columns))
    )
    df_super_matched = normalize_table(
        df_super_matched, query_category, question, query_super
    )

    try:
        assert_frame_equal(df_sub_normalized, df_super_matched, check_dtype=False)
        return True
    except AssertionError:
        return False


def compare_query_results(
    query_gold: str,
    query_gen: str,
    db_name: str,
    db_type: str,
    db_creds: dict,
    question: str,
    query_category: str,
    table_metadata_string: str = "",
    timeout: float = 10.0,
) -> "tuple[bool, bool]":
    """
    Compares the results of two queries and returns a tuple of booleans, where the first element is
    whether the queries produce exactly the same result, and the second element is whether the
    result of the gold query is a subset of the result of the generated query (still correct).
    We bubble up exceptions (mostly from query_postgres_db) to be handled in the runner.
    """
    query_gen = convert_postgres_to_sqlite3(query_gen)
    queries_gold = get_all_minimal_queries(query_gold)
    results_gen = query_sqllite3_db(query_gen, db_name, timeout)

    correct = False
    for q in queries_gold:
        q = convert_postgres_to_sqlite3(q)
        results_gold = query_sqllite3_db(q, db_name, timeout)
        
        if compare_df(
            results_gold, results_gen, query_category, question, query_gold, query_gen
        ):
            return (True, True)
        elif subset_df(results_gold, results_gen, query_category, question):
            correct = True
    return (False, correct)
