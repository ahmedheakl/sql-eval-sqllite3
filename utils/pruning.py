from typing import Dict, List, Optional
import numpy as np


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
