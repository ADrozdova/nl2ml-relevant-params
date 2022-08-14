import numpy as np

from transformers import RobertaTokenizer, RobertaModel

import torch
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaModel.from_pretrained("roberta_python_model")


def run_test_classification(data, classes, tokenizer, model):
    total_count = 0
    fnames_correct = 0
    doclong_correct = 0
    docline_correct = 0

    blocks_idx = np.unique(data.code_block_id)

    class_vec = []
    for c in classes.values():
        class_vec.append(model(tokenizer(c, return_tensors='pt')['input_ids'])[1])
    class_vec = torch.cat(class_vec, 0)

    class_idx = list(classes.keys())

    for cb_id in blocks_idx:
        functions = data.loc[data.code_block_id == cb_id]
        if len(functions) == 0:
            continue

        vertex_id = functions.graph_vertex_id.head(1).item()

        f_names = []
        docstrings_long = []
        docstrings_line = []
        docstrings_line_idx = []

        empty_docs = False

        for i in functions.index:
            if not isinstance(functions.loc[i].f_name, str):
                f_name = ""
            else:
                f_name = functions.loc[i].f_name

            if not isinstance(functions.loc[i].docstring, str) or functions.loc[i].docstring == "":
                doc_str = f_name  # ""
                empty_docs = True
            else:
                doc_str = functions.loc[i].docstring

            if len(doc_str) == 0:
                print(i, "len(docstr) == 0")

            f_name_vec = model(tokenizer(f_name, return_tensors='pt')['input_ids'])[1]

            scores = torch.softmax(torch.einsum("ab,cb->ac", f_name_vec, class_vec), -1)

            if ("Plot" in classes[vertex_id]) and ("Plot" in classes[class_idx[torch.argmax(scores).item()]]):
                f_names.append(1)
            else:
                f_names.append(classes[class_idx[torch.argmax(scores).item()]] == classes[vertex_id])

            long_end = 1000
            if len(doc_str[:1000]) == 1000:
                long_end = doc_str[:1000].rfind(" ")

            docstrings_long_vec = model(tokenizer(doc_str[:long_end], return_tensors='pt')['input_ids'])[1]

            scores = torch.softmax(torch.einsum("ab,cb->ac", docstrings_long_vec, class_vec), -1)

            if ("Plot" in classes[vertex_id]) and ("Plot" in classes[class_idx[torch.argmax(scores).item()]]):
                docstrings_long.append(1)
            else:
                docstrings_long.append(classes[class_idx[torch.argmax(scores).item()]] == classes[vertex_id])

            end = -1
            if not isinstance(doc_str, str):
                continue
            if doc_str.find("Parameters") != -1:
                end = doc_str.find("Parameters")
            if doc_str.find(">>>") != -1:
                if doc_str.find(">>>") < end or end == -1:
                    end = doc_str.find(">>>")
            if end == -1:
                end = doc_str.find("\n")
            if end > 1000:
                end = long_end

            docstrings_line_vec = model(tokenizer(doc_str[:end], return_tensors='pt')['input_ids'])[1]

            scores = torch.softmax(torch.einsum("ab,cb->ac", docstrings_line_vec, class_vec), -1)
            docstrings_line_idx.append(class_idx[torch.argmax(scores).item()])

            if ("Plot" in classes[vertex_id]) and ("Plot" in classes[class_idx[torch.argmax(scores).item()]]):
                docstrings_line.append(1)
            else:
                docstrings_line.append(classes[class_idx[torch.argmax(scores).item()]] == classes[vertex_id])

        if not empty_docs:
            total_count += len(functions.is_match)
            fnames_correct += np.sum(f_names == np.array(functions.is_match))
            doclong_correct += np.sum(docstrings_long == np.array(functions.is_match))
            docline_correct += np.sum(docstrings_line == np.array(functions.is_match))

    if total_count == 0:
        print("total_count == 0")
    else:
        print("accuracies", fnames_correct / total_count, doclong_correct / total_count, docline_correct / total_count)


def run_test_codesearch(data, classes, tokenizer, model):
    ALPHA = 0.9

    total_count = 0
    fnames_correct = 0
    doclong_correct = 0
    docline_correct = 0

    blocks_idx = np.unique(data.code_block_id)
    for cb_id in blocks_idx:
        functions = data.loc[data.code_block_id == cb_id]
        if len(functions) == 0:
            continue
        query = classes[functions.graph_vertex_id.head(1).item()]
        query_vec = model(tokenizer(query, return_tensors='pt')['input_ids'])[1]

        f_names = []
        docstrings_long = []
        docstrings_line = []

        empty_docs = False

        for i in functions.index:
            if not isinstance(functions.loc[i].f_name, str):
                f_name = ""
            else:
                f_name = functions.loc[i].f_name

            if not isinstance(functions.loc[i].docstring, str) or functions.loc[i].docstring == "":
                doc_str = f_name  # ""
                empty_docs = True
            else:
                doc_str = functions.loc[i].docstring.lower()

            if len(doc_str) == 0:
                print(i, "len(docstr) == 0")

            f_names.append(model(tokenizer(f_name, return_tensors='pt')['input_ids'])[1])
            docstrings_long.append(model(tokenizer(doc_str[:1000], return_tensors='pt')['input_ids'])[1])
            docstrings_line.append(model(tokenizer(doc_str.split("\n")[0], return_tensors='pt')['input_ids'])[1])

        if not empty_docs:
            f_names_vecs = torch.cat(f_names, 0)
            docstrings_long_vecs = torch.cat(docstrings_long, 0)
            docstrings_line_vecs = torch.cat(docstrings_line, 0)

            total_count += len(functions.is_match)

            scores = torch.softmax(torch.einsum("ab,cb->ac", query_vec, f_names_vecs), -1)
            preds = 1 * (scores.detach().numpy() >= ALPHA * np.max(scores.detach().numpy()))

            fnames_correct += np.sum(preds == np.array(functions.is_match))

            scores = torch.softmax(torch.einsum("ab,cb->ac", query_vec, docstrings_long_vecs), -1)
            preds = 1 * (scores.detach().numpy() >= ALPHA * np.max(scores.detach().numpy()))

            doclong_correct += np.sum(preds == np.array(functions.is_match))

            scores = torch.softmax(torch.einsum("ab,cb->ac", query_vec, docstrings_line_vecs), -1)
            preds = 1 * (scores.detach().numpy() >= ALPHA * np.max(scores.detach().numpy()))

            docline_correct += np.sum(preds == np.array(functions.is_match))

    print("accuracies", fnames_correct / total_count, doclong_correct / total_count, docline_correct / total_count)
