import os
import shutil
import tempfile
from pyls.workspace import Document, Workspace
from pyls.plugins.references import pyls_references
from pyls.plugins.signature import pyls_signature_help
from pyls.plugins.hover import pyls_hover

import ast
import pandas as pd
import inspect
import sys
import numpy as np


def get_func_docstrings(notebook_cells):
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
            if "import" in line:
                try:
                    exec(line)
                except Exception as e:
                    print("Could not run:\n", line, "\n", e)
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

    tmp = tempfile.mkdtemp()
    workspace = Workspace(tmp, None)

    def create_file(name, content):
        fn = os.path.join(tmp, name)
        with open(fn, 'w') as f:
            f.write(content)
        workspace.put_document('file://' + fn, content, None)

    create_file('code.py', cells)

    DOC_URI = 'file://' + os.path.join(workspace.root_uri, 'code.py')
    doc = Document(DOC_URI, workspace)

    result = []  # {codeblock_id, f_name, docstring}
    parsed = ast.parse(cells)

    for node in ast.walk(parsed):
        if isinstance(node, ast.Call):

            line_call = cells_lines[node.lineno - 1][node.col_offset:node.end_col_offset]
            idx = line_call.rfind("(")

            position_ref = {'line': node.lineno - 1, 'character': idx - 1 + node.col_offset}
            position_sig = {'line': node.lineno - 1, 'character': idx + 1 + node.col_offset}
            name = line_call[:idx]

            docstr = ""
            try:
                docstr = eval('inspect.getdoc(' + name + ')')
            except Exception:
                # find source through pyls
                signatures = pyls_signature_help(doc, position_sig)['signatures']
                if len(signatures) > 0 and 'documentation' in signatures[0] and len(signatures[0]['documentation']) > 0:
                    docstr = signatures[0]['documentation']
                else:
                    try:
                        refs = pyls_references(doc, position_ref)
                        if len(refs) == 0:
                            print("NO REFS", line_call, "\t", name.split(".")[-1])
                        else:
                            uri = refs[0]['uri']
                        source = []
                        if uri.find('site-packages') >= 0:
                            uri_split = uri.split('/')
                            source = uri_split[uri_split.index('site-packages') + 1:]
                            source[-1] = source[-1].split(".")[0]
                        f_name = name.split(".")[-1]
                        try:
                            exec("from " + ".".join(source) + " import " + f_name)
                        except Exception as e:
                            pass
                        try:
                            exec("from " + ".".join(source[:-1]) + " import " + f_name)
                        except Exception as e:
                            pass
                        try:
                            docstr = eval('inspect.getdoc(' + f_name + ')')
                        except Exception as e:
                            if len(source) == 0 or source[0] == "pandas":
                                try:
                                    exec("import pandas")
                                    docstr = eval('inspect.getdoc(' + "pandas.Series." + f_name + ')')
                                except Exception as e:
                                    pass
                                try:
                                    exec("import pandas")
                                    docstr = eval('inspect.getdoc(' + "pandas.DataFrame." + f_name + ')')
                                except Exception as e:
                                    pass
                                try:
                                    exec("from matplotlib.axes import Axes")
                                    docstr = eval('inspect.getdoc(' + "Axes." + f_name + ')')
                                except Exception as e:
                                    pass
                            else:
                                pass
                    except Exception as e:
                        pass
            pos_idx = np.where(lengths > node.lineno - 1)[0][0]
            result.append({"code_block_id": cb_ids[pos_idx], "graph_vertex_id": gv_ids[pos_idx],
                           "marks": marks[pos_idx], "f_name": name, "docstring": docstr})
    shutil.rmtree(tmp)

    return result


def codeblocks_docstrings(df):
    notebooks_grouped = df.groupby("kaggle_id").groups

    all_docstrings = pd.DataFrame(columns=["code_block_id", "f_name", "docstring", "graph_vertex_id", "marks"])

    for idx in notebooks_grouped.keys():
        notebook = df.iloc[notebooks_grouped[idx]].reset_index(drop=True)
        new_docstrings = get_func_docstrings(notebook)
        all_docstrings = all_docstrings.append(new_docstrings, ignore_index=True)

    return all_docstrings
