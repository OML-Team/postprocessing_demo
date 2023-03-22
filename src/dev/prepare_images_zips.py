import argparse
import zipfile

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.const import (
    ID_COLUMN,
    IMAGE_ID_SUFFIX,
    IMPROVED_SUFFIX,
    PATHS_COLUMN,
    WITHOUT_CHANGE_FLAG_VALUE,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-df", type=str, help="Path to an input file for query data frame.")
    parser.add_argument("--gallery-df", type=str, help="Path to an input file for query data frame.")
    parser.add_argument("--output-zipfile", type=str, help="Path to the output zip file.")
    parser.add_argument("--output-query-df", type=str, help="Path to an output file for query data frame.")
    parser.add_argument("--output-gallery-df", type=str, help="Path to an output file for gallery data frame.")
    return parser.parse_args()


def main():
    args = parse_args()
    query_df = pd.read_csv(args.query_df, index_col=0)
    gallery_df = pd.read_csv(args.gallery_df, index_col=0)

    improvement_columns_names = [c for c in query_df.columns if c.endswith(IMPROVED_SUFFIX)]
    diff_flags = query_df[improvement_columns_names[0]] != WITHOUT_CHANGE_FLAG_VALUE
    for column_name in improvement_columns_names[1:]:
        diff_flags = np.logical_or(diff_flags, query_df[column_name] != WITHOUT_CHANGE_FLAG_VALUE)
    filtered_query_df = query_df[diff_flags]
    filtered_query_df.to_csv(args.output_query_df)

    filtered_gallery_ids = set()
    for column_name in filtered_query_df.columns:
        if column_name.endswith(IMAGE_ID_SUFFIX):
            filtered_gallery_ids.update(filtered_query_df[column_name])
    filtered_gallery_df = gallery_df[gallery_df[ID_COLUMN].isin(filtered_gallery_ids)]
    filtered_gallery_df.to_csv(args.output_gallery_df)

    filepaths = set(filtered_query_df[PATHS_COLUMN])
    filepaths.update(filtered_gallery_df[PATHS_COLUMN])
    with zipfile.ZipFile(args.output_zipfile, "w", zipfile.ZIP_DEFLATED) as zipf:
        for filepath in tqdm(filepaths):
            zipf.write(filepath)


if __name__ == "__main__":
    main()
