import ast
import pandas as pd
import numpy as np
import re


def get_func_params(notebook_cells):
    cells = []
    lengths = []
    for cell in notebook_cells.code_block:
        format_ = f'b"'
        format_hat = f"b'"
        if cell[:2] == format_ or cell[:2] == format_hat:
            cell = eval(cell)
            cell = cell.decode('utf-8')

        cell = cell.replace("<br>", "\n")

        if cell.find("%matplotlib inline") > -1:
            cell = cell.replace("%matplotlib inline", "")

        cell = cell.strip()

        code = cell
        for line in cell.split("\n"):
            if len(line.strip()) > 0 and line.strip()[0] == "!":
                code = code.replace(line, "")
            if line.find("%time") > -1:
                line_ = line.replace("%time", "").strip()
                code = code.replace(line, line_)

        cells.append(code)
        lengths.append(len(code.split("\n")))

    lengths = np.cumsum(lengths)
    cb_ids = list(notebook_cells.code_block_id)
    gv_ids = list(notebook_cells.graph_vertex_id)
    marks = list(notebook_cells.marks)

    cells = "\n".join(cells)
    cells_lines = cells.split("\n")

    result = []  # {codeblock_id, f_name, params}
    parsed = ast.parse(cells)

    for node in ast.walk(parsed):
        if isinstance(node, ast.Call):
            params = []

            line_call = cells_lines[node.lineno - 1][node.col_offset:node.end_col_offset]
            idx = line_call.rfind("(")


            arg_1 = line_call.split(".")[0]
            arg = arg_1.lower()

            LIB_NAMES= ["pd", "pandas", "numpy", "np", "plt", "sklearn", "torch", "tf"]
            WHITELIST = ["df", "train", "test", "ax", "x", "y"]

            name = line_call[:idx]
            pos_idx = np.where(lengths > node.lineno - 1)[0][0]

            def check_whitelist(arg, wl):
                if name.split(".")[-1] == 'predict': #  models can have too many different names. class: model predict
                    return True
                if name.split(".")[-1] == 'fit' and gv_ids[pos_idx] == 26:  # class: model fit
                    return True
                for word in wl:
                    if word in arg:
                        return True
                return False

            if arg_1 not in LIB_NAMES:
                if check_whitelist(arg, WHITELIST):
                    params.append(arg_1)

            for arg in node.args:
                params.append(cells_lines[arg.lineno - 1][arg.col_offset:arg.end_col_offset])
            for keyw in node.keywords:
                keyw_arg = cells_lines[keyw.lineno - 1][keyw.col_offset:keyw.end_col_offset].split("=")[-1]
                if name.split(".")[-1] == 'drop' and keyw.arg == "columns":
                    params.append(keyw_arg)
                if name.split(".")[-1] == 'dropna' and keyw.arg == "subset":
                    params.append(keyw_arg)
                if name.split(".")[-1] == 'barplot' and (keyw.arg in ["x", "y", "data"]):
                    params.append(keyw_arg)

            result.append({"code_block_id": cb_ids[pos_idx], "graph_vertex_id": gv_ids[pos_idx],
                           "marks": marks[pos_idx], "f_name": name, "params": params})

    return result


def codeblocks_params(df):
    notebooks_grouped = df.groupby("kaggle_id").groups

    all_params = pd.DataFrame(columns=["code_block_id", "f_name", "params", "graph_vertex_id", "marks"])

    for idx in notebooks_grouped.keys():
        notebook = df.iloc[notebooks_grouped[idx]].reset_index(drop=True)
        new_params = get_func_params(notebook)
        all_params = all_params.append(new_params, ignore_index=True)

    return all_params


def get_all_params(notebook_cells):
    cells = []
    lengths = []
    for cell in notebook_cells.code_block:
        format_ = f'b"'
        format_hat = f"b'"
        if cell[:2] == format_ or cell[:2] == format_hat:
            cell = eval(cell)
            cell = cell.decode('utf-8')

        cell = cell.replace("<br>", "\n")

        if cell.find("%matplotlib inline") > -1:
            cell = cell.replace("%matplotlib inline", "")

        cell = cell.strip()

        code = cell
        for line in cell.split("\n"):
            if len(line.strip()) > 0 and line.strip()[0] == "!":
                code = code.replace(line, "")
            if line.find("%time") > -1:
                line_ = line.replace("%time", "").strip()
                code = code.replace(line, line_)

        cells.append(code)
        lengths.append(len(code.split("\n")))

    lengths = np.cumsum(lengths)
    cb_ids = list(notebook_cells.code_block_id)
    gv_ids = list(notebook_cells.graph_vertex_id)
    marks = list(notebook_cells.marks)

    cells = "\n".join(cells)
    cells_lines = cells.split("\n")

    result = []  # {codeblock_id, f_name, params}
    parsed = ast.parse(cells)

    for node in ast.walk(parsed):
        if isinstance(node, ast.Call):
            params = []

            line_call = cells_lines[node.lineno - 1][node.col_offset:node.end_col_offset]
            idx = line_call.rfind("(")

            params.append(line_call.split(".")[0])

            name = line_call[:idx]
            pos_idx = np.where(lengths > node.lineno - 1)[0][0]

            for arg in node.args:
                params.append(cells_lines[arg.lineno - 1][arg.col_offset:arg.end_col_offset])
            for keyw in node.keywords:
                keyw_arg = cells_lines[keyw.lineno - 1][keyw.col_offset:keyw.end_col_offset].split("=")[-1]
                params.append(keyw_arg)

            result.append({"code_block_id": cb_ids[pos_idx], "graph_vertex_id": gv_ids[pos_idx],
                           "marks": marks[pos_idx], "f_name": name, "params": params})

    return result


def codeblocks_params_all(df):
    notebooks_grouped = df.groupby("kaggle_id").groups

    all_params = pd.DataFrame(columns=["code_block_id", "f_name", "params", "graph_vertex_id", "marks"])

    for idx in notebooks_grouped.keys():
        notebook = df.iloc[notebooks_grouped[idx]].reset_index(drop=True)
        new_params = get_all_params(notebook)
        all_params = all_params.append(new_params, ignore_index=True)

    return all_params
