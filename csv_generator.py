import argparse
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from numpy import ndarray
from omegaconf import OmegaConf
from oml.const import (
    CATEGORIES_KEY,
    IS_GALLERY_KEY,
    IS_QUERY_KEY,
    LABELS_KEY,
    PATHS_KEY,
)
from oml.datasets.base import get_retrieval_datasets
from oml.functional.metrics import calc_cmc, calc_map
from oml.interfaces.models import IExtractor
from oml.interfaces.retrieval import IDistancesPostprocessor
from oml.registry.models import get_extractor_by_cfg
from oml.registry.postprocessors import get_postprocessor_by_cfg
from oml.registry.transforms import get_transforms_by_cfg
from oml.utils.misc_torch import pairwise_dist
from torch.utils.data import DataLoader
from tqdm import tqdm

from const import (
    CATEGORIES_COLUMN,
    CMC_IMPROVED_COLUMN_TEMPLATE,
    CMC_TOP_K_COLUMN_TEMPLATE,
    ID_COLUMN,
    IMPROVEMENT_FLAG_VALUE,
    LABELS_COLUMN,
    MAP_IMPROVED_COLUMN_TEMPLATE,
    MAP_TOP_K_COLUMN_TEMPLATE,
    PATHS_COLUMN,
    POSTPROCESSED_CMC_TOP_K_COLUMN_TEMPLATE,
    POSTPROCESSED_MAP_TOP_K_COLUMN_TEMPLATE,
    POSTPROCESSED_TOP_K_IMAGE_ID_COLUMN_TEMPLATE,
    POSTPROCESSED_TOP_K_SCORE_COLUMN_TEMPLATE,
    SIMPLE_IMPROVED_COLUMN,
    TOP_K_IMAGE_ID_COLUMN_TEMPLATE,
    TOP_K_SCORE_COLUMN_TEMPLATE,
    WITHOUT_CHANGE_FLAG_VALUE,
    WORSENING_FLAG_VALUE,
)

torch.multiprocessing.set_sharing_strategy("file_system")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extractor-cfg", type=str, help="Path to the extractor config.")
    parser.add_argument("--postprocessor-cfg", type=str, help="Path to the extractor config.")
    parser.add_argument("--max-top-k", type=int, help="Maximal number of top-k to consider.")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda", help="Device type.")
    parser.add_argument("--embeddings-filepath", type=str, help="Path to a file where embeddings will be saved to.")
    parser.add_argument("--query-df", type=str, help="Path to a file for query data frame.")
    parser.add_argument("--gallery-df", type=str, help="Path to a file for query data frame.")
    parser.add_argument("--cmc-top-ks", type=int, nargs="+", help="Top-k to calculate CMC@k for.")
    parser.add_argument("--map-top-ks", type=int, nargs="+", help="Top-k to calculate MAP@k for.")
    return parser.parse_args()


def main():
    args = parse_args()
    max_top_k = args.max_top_k
    max_top_k = max([max_top_k, *args.cmc_top_ks, *args.map_top_ks])
    device = args.device
    extractor_cfg = OmegaConf.load(args.extractor_cfg)
    postprocessor_cfg = OmegaConf.load(args.postprocessor_cfg)
    extractor_transforms_val = get_transforms_by_cfg(extractor_cfg["transforms_val"])
    extractor = get_extractor_by_cfg(extractor_cfg["model"])
    extractor.to(device)
    extractor.eval()

    postprocessor = get_postprocessor_by_cfg(postprocessor_cfg["postprocessor"])
    postprocessor.model.to(device)
    postprocessor.model.eval()

    _, valid_dataset = get_retrieval_datasets(
        dataset_root=Path(extractor_cfg["dataset_root"]),
        transforms_train=None,
        transforms_val=extractor_transforms_val,
        dataframe_name=extractor_cfg["dataframe_name"],
    )
    loader_valid = DataLoader(
        dataset=valid_dataset,
        batch_size=extractor_cfg["bs_val"],
        num_workers=extractor_cfg["num_workers"],
        drop_last=False,
        shuffle=False,
    )
    embeddings, labels, is_query, is_gallery, categories, paths, ids = extract_embeddings(extractor, loader_valid)
    np.savez(
        args.embeddings_filepath,
        embeddings=embeddings,
        labels=labels,
        is_query=is_query,
        is_gallery=is_gallery,
        categories=categories,
        paths=paths,
        ids=ids,
    )
    # data = np.load("data/test_dataset/valid_features_big.npz")
    # embeddings = data["embeddings"]
    # is_query = data["is_query"]
    # is_gallery = data["is_gallery"]
    # categories = data["categories"]
    # paths = data["paths"]
    # ids = data["ids"]

    query_df, gallery_df = eval_dataframe(
        embeddings,
        ids,
        is_gallery,
        is_query,
        categories,
        labels,
        paths,
        postprocessor,
        device,
        max_top_k,
        args.cmc_top_ks,
        args.map_top_ks,
    )
    gallery_df.to_csv(args.gallery_df)
    query_df.to_csv(args.query_df)


def extract_embeddings(
    extractor: IExtractor, data_loader: DataLoader
) -> Tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray, ndarray]:
    device = next(extractor.model.parameters()).device
    embeddings = []
    paths = []
    labels = []
    is_query = []
    is_gallery = []
    categories = []
    with torch.no_grad():
        for batch in tqdm(data_loader, leave=False):
            embeddings.append(extractor.forward(batch["input_tensors"].to(device)).cpu().numpy())
            labels.append(batch[LABELS_KEY].numpy())
            paths.append(batch[PATHS_KEY])
            if CATEGORIES_KEY in batch:
                categories.append(batch[CATEGORIES_KEY])
            if IS_QUERY_KEY in batch:
                is_query.append(batch[IS_QUERY_KEY].numpy())
            if IS_GALLERY_KEY in batch:
                is_gallery.append(batch[IS_GALLERY_KEY].numpy())

    embeddings = np.concatenate(embeddings)
    labels = np.concatenate(labels)
    paths = np.concatenate(paths)
    ids = np.arange(len(embeddings))
    categories = np.concatenate(categories) if categories else np.zeros((0,))
    is_query = np.concatenate(is_query) if is_query else np.zeros((0,))
    is_gallery = np.concatenate(is_gallery) if is_gallery else np.zeros((0,))
    return embeddings, labels, is_query, is_gallery, categories, paths, ids


def eval_dataframe(
    embeddings: np.ndarray,
    ids: np.ndarray,
    is_gallery: np.ndarray,
    is_query: np.ndarray,
    categories: np.ndarray,
    labels: np.ndarray,
    paths: np.ndarray,
    postprocessor: IDistancesPostprocessor,
    device: str,
    max_top_k: int,
    cmc_top_ks: List[int],
    map_top_ks: List[int],
):
    embeddings = torch.from_numpy(embeddings).to(device)
    query_inds = np.where(is_query)[0]
    gallery_ids = ids[is_gallery]
    gallery_embeddings = embeddings[is_gallery, :]
    gallery_paths = paths[is_gallery]
    gallery_categories = categories[is_gallery]
    gallery_labels = labels[is_gallery]
    gallery_df = pd.DataFrame(columns=[ID_COLUMN, PATHS_COLUMN, CATEGORIES_COLUMN, LABELS_COLUMN])
    gallery_df[ID_COLUMN] = gallery_ids
    gallery_df[PATHS_COLUMN] = gallery_paths
    gallery_df[CATEGORIES_COLUMN] = gallery_categories
    gallery_df[LABELS_COLUMN] = gallery_labels
    df = defaultdict(list)
    for i in tqdm(query_inds):
        df[ID_COLUMN].append(ids[i])
        df[PATHS_COLUMN].append(paths[i])
        df[CATEGORIES_COLUMN].append(categories[i])
        df[LABELS_COLUMN].append(labels[i])
        embedding = embeddings[i, :]

        distances = pairwise_dist(embedding.reshape(1, -1), gallery_embeddings)
        distances[0, gallery_ids == ids[i]] = torch.inf

        def _eval_metrics(_distances, column_names):
            top_k_inds = torch.topk(_distances, k=max_top_k, largest=False)[1]
            matching_result = gallery_labels[top_k_inds.cpu().numpy()] == labels[i]
            matching_result = torch.from_numpy(matching_result).to(device).view(1, -1)
            cmc = [c.cpu().numpy().item() for c in calc_cmc(matching_result, cmc_top_ks)]
            n_gt = torch.tensor(sum(gallery_labels == labels[i])).to(device).view(1, 1)
            map_ = [m.cpu().numpy().item() for m in calc_map(matching_result, n_gt, map_top_ks)]
            scores = np.squeeze(_distances[0, top_k_inds].cpu().numpy())
            for j, top_id in enumerate(top_k_inds[0, :], start=1):
                df[column_names[0] % j].append(scores[j - 1])
                df[column_names[1] % j].append(gallery_ids[top_id.item()])
            for j, c in zip(cmc_top_ks, cmc):
                df[column_names[2] % j].append(c)
            for j, m in zip(map_top_ks, map_):
                df[column_names[3] % j].append(m)

            return cmc, map_, matching_result

        cmc_before_stir, map_before_stir, matching_result_before_stir = _eval_metrics(
            distances,
            [
                TOP_K_SCORE_COLUMN_TEMPLATE,
                TOP_K_IMAGE_ID_COLUMN_TEMPLATE,
                CMC_TOP_K_COLUMN_TEMPLATE,
                MAP_TOP_K_COLUMN_TEMPLATE,
            ],
        )

        with torch.no_grad():
            distances = postprocessor.process(distances, [paths[i]], gallery_paths)

        cmc_after_stir, map_after_stir, matching_result_after_stir = _eval_metrics(
            distances,
            [
                POSTPROCESSED_TOP_K_SCORE_COLUMN_TEMPLATE,
                POSTPROCESSED_TOP_K_IMAGE_ID_COLUMN_TEMPLATE,
                POSTPROCESSED_CMC_TOP_K_COLUMN_TEMPLATE,
                POSTPROCESSED_MAP_TOP_K_COLUMN_TEMPLATE,
            ],
        )
        matching_result_before_stir = np.squeeze(matching_result_before_stir.cpu().numpy()).tolist()
        matching_result_after_stir = np.squeeze(matching_result_after_stir.cpu().numpy()).tolist()
        for column_name_template, metric_before_stir, metric_after_stir, top_ks in [
            [CMC_IMPROVED_COLUMN_TEMPLATE, cmc_before_stir, cmc_after_stir, cmc_top_ks],
            [MAP_IMPROVED_COLUMN_TEMPLATE, map_before_stir, map_after_stir, map_top_ks],
        ]:
            for m, mp, k in zip(metric_before_stir, metric_after_stir, top_ks):
                if m == mp:
                    df[column_name_template % k].append(WITHOUT_CHANGE_FLAG_VALUE)
                elif mp > m:
                    df[column_name_template % k].append(IMPROVEMENT_FLAG_VALUE)
                else:
                    df[column_name_template % k].append(WORSENING_FLAG_VALUE)
        if matching_result_before_stir == matching_result_after_stir:
            df[SIMPLE_IMPROVED_COLUMN].append(WITHOUT_CHANGE_FLAG_VALUE)
        elif matching_result_after_stir > matching_result_before_stir:
            df[SIMPLE_IMPROVED_COLUMN].append(IMPROVEMENT_FLAG_VALUE)
        else:
            df[SIMPLE_IMPROVED_COLUMN].append(WORSENING_FLAG_VALUE)
    query_df = pd.DataFrame(df)
    return query_df, gallery_df


if __name__ == "__main__":
    main()
