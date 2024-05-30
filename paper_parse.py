import json
from collections import defaultdict, Counter
from itertools import combinations
import re
import processing_files
import os

# "blocks": [
#         {
#           "lines": [
#             {
#               "spans": [
#                 {
#                   "text": "Review\b",
#                   "font": "TimesNewRomanPSMT",
#                   "size": 12,
#                   "color": 0,
#                   "bbox": [
#                     205.35060119628906,
#                     57.15484619140625,
#                     244.5426025390625,
#                     70.44390869140625
#                   ]
#                 }
#               ]
#             },


def patch_bbox(dct):
    if isinstance(dct, dict):
        for v in dct.values():
            patch_bbox(v)
        if "bbox" in dct:
            dct["x"] = round(dct["bbox"][0], 2)
            dct["y"] = round(dct["bbox"][1], 2)
            dct["xx"] = round(dct["bbox"][2], 2)
            dct["yy"] = round(dct["bbox"][3], 2) # просто переименовываем для читаемости
        if "spans" in dct: # отчистка пустых
            dct["spans"] = [span for span in dct["spans"] if span["text"].strip() != ""]
        if "lines" in dct:
            dct["lines"] = [line for line in dct["lines"] if len(line["spans"]) > 0]
    if isinstance(dct, list):
        for d in dct:
            patch_bbox(d)
    return dct


def sorted_lines(block):
    return sorted(
        block["lines"],
        key=lambda line: min(span["bbox"][1] for span in line["spans"]),
    )  # for each line from top to bottom # сортируем все по y координтате


def sorted_spans(line):
    return sorted(
        line["spans"], key=lambda span: span["bbox"][0]
    )  # for each span from left to right # внутри текст по левости


def sorted_block_items(block):
    return [span for line in sorted_lines(block) for span in sorted_spans(line)]


def block_items(block):
    return [span for line in block["lines"] for span in line["spans"]]


def block_sort_top(blocks):
    return list(sorted(blocks, key=lambda item: (item["y"], item["x"])))


def block_sort_left(blocks):
    return list(sorted(blocks, key=lambda item: (item["x"], item["y"])))


def get_left(b1, b2):
    return (b1, b2) if b1["x"] <= b2["x"] else (b2, b1)


def intersect_x(b1, b2):
    b1, b2 = get_left(b1, b2)
    return b1["xx"] > b2["x"]


def get_top(b1, b2):
    return (b1, b2) if b1["y"] <= b2["y"] else (b2, b1)


def intersect_y(b1, b2):
    b1, b2 = get_top(b1, b2)
    return b1["yy"] > b2["y"]


def intersect_area_y(b1, b2):
    b1, b2 = get_top(b1, b2)
    if b1["yy"] <= b2["y"]:
        return 0
    return (b1["yy"] - b2["y"]) / (b2["yy"] - b1["y"])


def dist_y(b1, b2):
    b1, b2 = get_top(b1, b2)
    return max(b2["y"] - b1["yy"], -1e6)


def dist_x(b1, b2):
    b1, b2 = get_left(b1, b2)
    return max(b2["x"] - b1["xx"], -1e6)


def is_merge_blocks(b1, b2, x_dist=0, y_dist=0):
    cond = False
    cond |= intersect_x(b1, b2) and intersect_y(b1, b2)
    cond |= intersect_x(b1, b2) and dist_y(b1, b2) <= max(y_dist, CONST_MERGE)
    cond |= intersect_y(b1, b2) and dist_x(b1, b2) <= max(x_dist, CONST_MERGE)
    return cond


def make_set(x):
    x["parent"] = None


def find_set(x):
    return x if x["parent"] is None else find_set(x["parent"])


def merge_set(a, b, y_min=0):
    a = find_set(a)
    b = find_set(b)
    if a == b:
        return
    b["parent"] = a

    x = a
    if a["y"] != b["y"]:
        aa, bb = get_top(a, b)
        top_text, bot_text = aa["text"], bb["text"]
        if bb["y"] - aa["yy"] >= y_min:
            if aa["x"] > bb["x"] + PAR_THRESHOLD:  # new paragraph??
                top_text = "<PAR>" + top_text
            elif bb["x"] > aa["x"] + PAR_THRESHOLD:
                bot_text = "<PAR>" + bot_text
        x["text"] = NEW_LINE.join([top_text, bot_text])
    else:
        aa, bb = get_left(a, b)
        x["text"] = "".join([aa["text"], bb["text"]])

    # merge attributes
    x["x"] = min(a["x"], b["x"])
    x["y"] = min(a["y"], b["y"])
    x["xx"] = max(a["xx"], b["xx"])
    x["yy"] = max(a["yy"], b["yy"])

    x["text_size"] = a["text_size"] + b["text_size"]


def find_abstract_name(a, blocks, max_dist=5):
    return any(
        [
            b["text"].strip().lower() == "abstract"
            and max_dist >= a["y"] - b["yy"] >= 0
            for b in blocks
        ]
    )


def try_abstract(blocks, ABSTRACT_COMMONS, font_ratio=0.95, min_size=50, max_dist=15):

    for block in blocks:
        if (
            any(
                block["text_size"].get(abstract_font, 0)
                / sum(block["text_size"].values())
                >= font_ratio
                for abstract_font in ABSTRACT_COMMONS
            )
            and len(block["text"].strip().split(" ")) >= min_size
            and (
                block["text"].replace("<PAR>", "").lower().startswith("abstract")
                or find_abstract_name(block, blocks, max_dist=max_dist)
            )
        ):
            return [block]

    return []

# add <PAR> token to first block and for each block separated by space
def prepend_par(b):
    if not b["text"].startswith("<PAR>"):
        b["text"] = f"<PAR>{b['text']}"

def clean_text(text):
    text = re.sub(r'\s*­\s*', '', text)
    text = re.sub(r'\s*-\s*', '', text)

    text = re.sub(r'\s{2,}', ' ', text)

    return text
import sys

NUM_PAGES = 3
CONST_MERGE = 2.0
PAR_THRESHOLD = 1.0
NEW_LINE = " "

def parse(file_name):
    # file_name = "Papers/results (11)/4972757751301391959.json"
    with open(file_name, "r", encoding="utf-8") as f:
        data = json.load(f)


    pages = [
        patch_bbox(page)
        for page in data["pages"]
        if page["page_number"] in range(1, NUM_PAGES + 1)
    ]

    LINE_DISTS = Counter([0])
    SPAN_DISTS = Counter([0])
    TEXT_SIZES = Counter([])
    for page in pages:
        for block in page["blocks"]:
            lines = list(sorted_lines(block))
            LINE_DISTS += Counter(
                [
                    min(span["y"] for span in line2["spans"])
                    - max(span["yy"] for span in line1["spans"])
                    for line1, line2 in zip(lines[:-1], lines[1:])
                    if intersect_area_y(
                        {
                            "y": min(span["y"] for span in line1["spans"]),
                            "yy": max(span["yy"] for span in line1["spans"]),
                        },
                        {
                            "y": min(span["y"] for span in line2["spans"]),
                            "yy": max(span["yy"] for span in line2["spans"]),
                        },
                    )
                    <= 0.5
                ]
            )
            SPAN_DISTS += Counter(
                [
                    span2["x"] - span1["xx"]
                    for line in lines
                    for span1, span2 in zip(
                        list(sorted_spans(line))[:-1], list(sorted_spans(line))[1:]
                    )
                ]
            )
            TEXT_SIZES += Counter(
                [
                    f"{span['size']}_{span['font']}"
                    for line in lines
                    for span in line["spans"]
                    for sym in span["text"].strip()
                ]
            )

    X_DIST, x_count = SPAN_DISTS.most_common(1)[0]
    Y_DIST, y_count = LINE_DISTS.most_common(1)[0]

    # for overlapping papers
    X_MIN = min(SPAN_DISTS.keys())
    Y_MIN = min(LINE_DISTS.keys())

    TEXT_COMMON, text_count = TEXT_SIZES.most_common(1)[0]
    ABSTRACT_COMMONS = [item[0] for item in TEXT_SIZES.most_common(3)]


    total_blocks = []
    abstract_found = False
    for page_idx, page in enumerate(pages):
        blocks = []
        for block_idx, block in enumerate(page["blocks"]):
            lines = list(sorted_lines(block))
            for line in lines:
                item = {
                    "idx": int(page_idx * 1e6 + block_idx),
                    "x": min(span["x"] for span in line["spans"]),
                    "y": min(span["y"] for span in line["spans"]),
                    "xx": max(span["xx"] for span in line["spans"]),
                    "yy": max(span["yy"] for span in line["spans"]),
                    "text_size": Counter(
                        [
                            f"{span['size']}_{span['font']}"
                            for span in line["spans"]
                            for sym in span["text"].strip()
                        ]
                    ),
                    "text": " ".join(span["text"] for span in sorted_spans(line)),
                    "page": page_idx,
                }
                blocks.append(item)

        # merging close blocks by iterating over all pairs
        [make_set(block) for block in block_sort_top(blocks)]
        for i in range(40):
            num_merges = 0
            for b1, b2 in combinations(blocks, 2):
                if is_merge_blocks(b1, b2, x_dist=X_DIST, y_dist=Y_DIST):
                    merge_set(b1, b2, y_min=Y_MIN)
                    num_merges += 1
            blocks = [block for block in blocks if block["parent"] is None]
            if num_merges == 0:
                break
        # DEBUG
        # with open(f"merged_{page_idx}.json", "w", encoding="utf-8") as f:
        #     json.dump(blocks, f)

        # filter by text size
        FILTER_RATIO = 0.85
        abstract = try_abstract(blocks, ABSTRACT_COMMONS=ABSTRACT_COMMONS)
        abstract_found |= len(abstract) > 0
        blocks = [
            block
            for block in blocks
            if block["text_size"].get(TEXT_COMMON, 0) / sum(block["text_size"].values())
            >= FILTER_RATIO
            and block["idx"] not in [a["idx"] for a in abstract]
        ]

        # if page_idx == 0:
        #     # get abstract as most top item
        #     blocks = block_sort_top(blocks)
        #     abstract, blocks = blocks[:1], blocks[1:]
        #     # sort other by x -> y without abstract
        #     blocks = abstract + block_sort_left(blocks)
        # else:
        total_blocks.extend(abstract + block_sort_left(blocks))

    # add <PAR> token to first block and for each block separated by space
    # def prepend_par(b):
    #     if not b["text"].startswith("<PAR>"):
    #         b["text"] = f"<PAR>{b['text']}"


    for block_idx, block in enumerate(total_blocks):
        block["text"] = block["text"].strip()
        prev_block = total_blocks[block_idx - 1]
        if block_idx == 0 or (
            intersect_x(prev_block, block) and prev_block["page"] == block["page"]
        ):
            prepend_par(block)

    # join all blocks with new line
    final_text = NEW_LINE.join(block["text"] for block in total_blocks)

    # Merge too small paragraphs to previous paragraph
    __paragraphs = [par for par in final_text.split("<PAR>") if par.strip() != ""]
    paragraphs = []
    MIN_WORDS = 30
    for par in __paragraphs:
        if len(par.strip().split(" ")) < MIN_WORDS and len(paragraphs) > 0:
            paragraphs[-1] += f"{NEW_LINE}{par}"
        else:
            paragraphs.append(par)

    final_text = "<PAR>".join(paragraphs)
    # print(f"FOUND ABSTRACT {abstract_found}")


    final_text = clean_text(final_text)
    # print(final_text)

    return abstract_found, final_text

    # remaining_files, removed_files = processing_files.delete_files("Papers")
    # print(f"Число оставшихся файлов: {remaining_files}")
    # print(f"Количество удаленных файлов: {removed_files}")