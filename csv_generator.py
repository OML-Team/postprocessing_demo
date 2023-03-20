import argparse
import itertools
from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from oml.const import (
    CATEGORIES_COLUMN,
    IS_GALLERY_COLUMN,
    IS_QUERY_COLUMN,
    LABELS_COLUMN,
    PATHS_COLUMN,
    SPLIT_COLUMN,
)
from oml.functional.metrics import apply_mask_to_ignore, calc_cmc, calc_map
from oml.inference.flat import inference_on_images
from oml.inference.pairs import pairwise_inference_on_images
from oml.interfaces.retrieval import IDistancesPostprocessor
from oml.registry.models import get_extractor_by_cfg
from oml.registry.postprocessors import get_postprocessor_by_cfg
from oml.registry.transforms import get_transforms_by_cfg
from oml.utils.misc_torch import assign_2d
from sklearn.neighbors import NearestNeighbors

from const import (
    CMC_IMPROVED_COLUMN_TEMPLATE,
    CMC_TOP_K_COLUMN_TEMPLATE,
    ID_COLUMN,
    IMPROVEMENT_FLAG_VALUE,
    MAP_IMPROVED_COLUMN_TEMPLATE,
    MAP_TOP_K_COLUMN_TEMPLATE,
    POSTPROCESSED_CMC_TOP_K_COLUMN_TEMPLATE,
    POSTPROCESSED_MAP_TOP_K_COLUMN_TEMPLATE,
    POSTPROCESSED_TOP_K_IMAGE_ID_COLUMN_TEMPLATE,
    POSTPROCESSED_TOP_K_SCORE_COLUMN_TEMPLATE,
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
    parser.add_argument("--df-filepath", type=str, help="Path to the dataset dataframe.")
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
    df = pd.read_csv(args.df_filepath, index_col=False)
    df = df[df[SPLIT_COLUMN] == "validation"]
    extractor_cfg = OmegaConf.load(args.extractor_cfg)
    extractor = get_extractor_by_cfg(extractor_cfg["model"])
    extractor.to(device)
    extractor.eval()
    extractor_transforms_val = get_transforms_by_cfg(extractor_cfg["transforms_val"])

    postprocessor_cfg = OmegaConf.load(args.postprocessor_cfg)
    postprocessor = get_postprocessor_by_cfg(postprocessor_cfg["postprocessor"])
    postprocessor.model.to(device)
    postprocessor.model.eval()

    embeddings = inference_on_images(
        model=extractor,
        paths=df[PATHS_COLUMN].tolist(),
        transform=extractor_transforms_val,
        batch_size=extractor_cfg.bs_val,
        num_workers=extractor_cfg.num_workers,
        verbose=True,
    )
    np.savez(
        args.embeddings_filepath,
        embeddings=embeddings,
    )
    # data = np.load("data/SOP/embeddings.npz")
    # embeddings = data["embeddings"]
    # is_query = data["is_query"]
    # is_gallery = data["is_gallery"]
    # categories = data["categories"]
    # paths = data["paths"]
    # ids = data["ids"]

    query_df, gallery_df = eval_dataframe(
        embeddings,
        postprocessor,
        df,
        device,
        max_top_k,
        args.cmc_top_ks,
        args.map_top_ks,
    )
    gallery_df.to_csv(args.gallery_df)
    query_df.to_csv(args.query_df)


def eval_dataframe(
    embeddings: np.ndarray,
    postprocessor: IDistancesPostprocessor,
    df: pd.DataFrame,
    device: str,
    max_top_k: int,
    cmc_top_ks: List[int],
    map_top_ks: List[int],
):
    is_gallery = (df[IS_GALLERY_COLUMN] == 1).to_numpy()
    is_query = (df[IS_QUERY_COLUMN] == 1).to_numpy()
    gallery_ids = df.index[is_gallery].to_numpy()
    gallery_labels = df[is_gallery][LABELS_COLUMN].to_numpy()
    query_ids = df.index[is_query].to_numpy()
    query_labels = df[is_query][LABELS_COLUMN].to_numpy()
    gallery_paths = df[is_gallery][PATHS_COLUMN].to_numpy()
    query_paths = df[is_query][PATHS_COLUMN].tolist()
    gallery_df = pd.DataFrame(columns=[ID_COLUMN, PATHS_COLUMN, CATEGORIES_COLUMN, LABELS_COLUMN])
    gallery_df[ID_COLUMN] = gallery_ids
    gallery_df[PATHS_COLUMN] = gallery_paths
    gallery_df[CATEGORIES_COLUMN] = df[is_gallery][CATEGORIES_COLUMN]
    gallery_df[LABELS_COLUMN] = gallery_labels

    knn = NearestNeighbors()
    knn.fit(embeddings[is_gallery, :])
    distances, ii_top = knn.kneighbors(embeddings[is_query, :], n_neighbors=max_top_k + 1)
    distances = torch.from_numpy(distances)
    mask_gt = torch.from_numpy(gallery_labels[ii_top] == query_labels[:, np.newaxis])
    # it is done like that in order to prevent OOM for huge datasets
    n_gt = torch.tensor([np.sum(ql == gallery_labels) for ql in query_labels]).view(-1, 1)
    mask_to_ignore = torch.from_numpy(gallery_ids[ii_top] == query_ids[:, np.newaxis])
    distances, mask_gt = apply_mask_to_ignore(distances, mask_gt, mask_to_ignore)
    sort_inds = torch.argsort(distances, dim=1)
    distances = torch.take_along_dim(distances, sort_inds, dim=1)
    ii_top = np.take_along_axis(ii_top, sort_inds.cpu().numpy(), axis=1)
    mask_gt = torch.take_along_dim(mask_gt, sort_inds, dim=1)

    query_df = pd.DataFrame()
    query_df[ID_COLUMN] = df.index[is_query].to_numpy()
    query_df[PATHS_COLUMN] = df[is_query][PATHS_COLUMN]
    query_df[LABELS_COLUMN] = df[is_query][LABELS_COLUMN]
    query_df[CATEGORIES_COLUMN] = df[is_query][CATEGORIES_COLUMN]

    cmc_before, map_before = eval_metrics(
        query_df,
        distances,
        mask_gt,
        ii_top,
        gallery_ids,
        n_gt,
        cmc_top_ks,
        map_top_ks,
        max_top_k,
        TOP_K_SCORE_COLUMN_TEMPLATE,
        TOP_K_IMAGE_ID_COLUMN_TEMPLATE,
        CMC_TOP_K_COLUMN_TEMPLATE,
        MAP_TOP_K_COLUMN_TEMPLATE,
    )

    with torch.no_grad():
        _ii_top = torch.from_numpy(ii_top[:, : postprocessor.top_n]).contiguous()
        _ii_top = torch.repeat_interleave(torch.arange(0, postprocessor.top_n).view(1, -1), distances.shape[0], dim=0)
        distances = process(postprocessor, distances, _ii_top, query_paths, gallery_paths)
    cmc_after, map_after = eval_metrics(
        query_df,
        distances,
        mask_gt,
        ii_top,
        gallery_ids,
        n_gt,
        cmc_top_ks,
        map_top_ks,
        max_top_k,
        POSTPROCESSED_TOP_K_SCORE_COLUMN_TEMPLATE,
        POSTPROCESSED_TOP_K_IMAGE_ID_COLUMN_TEMPLATE,
        POSTPROCESSED_CMC_TOP_K_COLUMN_TEMPLATE,
        POSTPROCESSED_MAP_TOP_K_COLUMN_TEMPLATE,
    )

    for column_name_template, metric_before_stir, metric_after_stir, top_ks in [
        [CMC_IMPROVED_COLUMN_TEMPLATE, cmc_before, cmc_after, cmc_top_ks],
        [MAP_IMPROVED_COLUMN_TEMPLATE, map_before, map_after, map_top_ks],
    ]:
        for m, mp, k in zip(metric_before_stir, metric_after_stir, top_ks):
            query_df[column_name_template % k] = WITHOUT_CHANGE_FLAG_VALUE
            query_df[(m > mp).cpu().numpy()] = IMPROVEMENT_FLAG_VALUE
            query_df[(m < mp).cpu().numpy()] = WORSENING_FLAG_VALUE
    return query_df, gallery_df


def eval_metrics(
    df,
    distances,
    mask_gt,
    top_k_inds,
    gallery_ids,
    n_gt,
    cmc_top_ks,
    map_top_ks,
    max_top_k,
    top_k_score_column_template,
    top_k_image_id_column_template,
    cmc_column_template,
    map_column_template,
):
    sort_inds = torch.argsort(distances, dim=1)
    distances = torch.take_along_dim(distances, sort_inds, dim=1)
    top_k_inds = np.take_along_axis(top_k_inds, sort_inds.cpu().numpy(), axis=1)
    mask_gt = torch.take_along_dim(mask_gt, sort_inds, dim=1)

    cmc_before = calc_cmc(mask_gt, cmc_top_ks)
    map_before = calc_map(mask_gt, n_gt, map_top_ks)

    for i in range(max_top_k):
        df[top_k_score_column_template % (i + 1)] = distances[:, i].cpu().numpy()
        df[top_k_image_id_column_template % (i + 1)] = gallery_ids[top_k_inds[:, i]]
    for k, c in zip(cmc_top_ks, cmc_before):
        df[cmc_column_template % k] = c.cpu().numpy()
    for k, m in zip(map_top_ks, map_before):
        df[map_column_template % k] = m.cpu().numpy()
    return cmc_before, map_before


def process(
    postprocessor: IDistancesPostprocessor, distances: torch.Tensor, ii_top: torch.Tensor, queries: Any, galleries: Any
) -> torch.Tensor:
    n_galleries = len(galleries)

    # 1. Adjust top_n with respect to the actual gallery size and find top-n pairs
    top_n = min(postprocessor.top_n, n_galleries)

    # 2. Create (n_queries * top_n) pairs of each query and related galleries and re-estimate distances for them
    if postprocessor.verbose:
        print("\nPostprocessor's inference has been started...")
    distances_upd = postprocessor.inference(queries=queries, galleries=galleries, ii_top=ii_top, top_n=top_n)
    distances_upd = distances_upd.to(distances.device).to(distances.dtype)

    # 3. Update distances for top-n galleries
    # The idea is that we somehow permute top-n galleries, but rest of the galleries
    # we keep in the end of the list as before permutation.
    # To do so, we add an offset to these galleries' distances (which haven't participated in the permutation)
    if top_n < n_galleries:
        # Here we use the fact that distances not participating in permutation start with top_n + 1 position
        min_in_old_distances = torch.topk(distances, k=top_n + 1, largest=False)[0][:, -1]
        max_in_new_distances = distances_upd.max(dim=1)[0]
        offset = max_in_new_distances - min_in_old_distances + 1e-5  # we also need some eps if max == min
        distances += offset.unsqueeze(-1)
    else:
        # Pairwise postprocessor has been applied to all possible pairs, so, there are no rest distances.
        # Thus, we don't need to care about order and offset at all.
        pass

    distances = assign_2d(x=distances, indices=ii_top, new_values=distances_upd)

    return distances


def inference(
    postprocessor, queries: List[Path], galleries: List[Path], ii_top: torch.Tensor, top_n: int
) -> torch.Tensor:
    n_queries = len(queries)
    queries = list(itertools.chain.from_iterable(itertools.repeat(x, top_n) for x in queries))
    galleries = [galleries[i] for i in ii_top.view(-1)]
    distances_upd = pairwise_inference_on_images(
        model=postprocessor.model,
        paths1=queries,
        paths2=galleries,
        transform=postprocessor.image_transforms,
        num_workers=postprocessor.num_workers,
        batch_size=postprocessor.batch_size,
        verbose=postprocessor.verbose,
        use_fp16=postprocessor.use_fp16,
    )
    distances_upd = distances_upd.view(n_queries, top_n)
    return distances_upd


if __name__ == "__main__":
    main()
