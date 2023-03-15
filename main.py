from collections import defaultdict
from typing import Dict, Union

import cv2
import streamlit as st

from const import (
    BORDER_SIZE,
    DATASETS,
    GREEN_COLOR,
    IMPROVED_SUFFIX,
    METRICS_TO_EXCLUDE_FROM_VIEWER,
    SIZE,
    TOP_K,
    ImprovementFlags,
    RetrievalResultsType,
)
from controls import GalleryViewer, QueryViewer
from data import (
    GalleryDataset,
    QueryDataset,
    QuerySample,
    load_gallery_dataset,
    load_query_dataset,
)
from utils import pad_image_to_square

st.set_page_config(layout="wide", page_title="similarity-api")


def main():
    datasets = download_datasets(DATASETS)
    st.sidebar.subheader("Dataset")
    dataset_name = st.sidebar.selectbox("Dataset", datasets, label_visibility="collapsed")
    query_dataset = load_query_dataset(datasets[dataset_name]["query"], datasets[dataset_name]["zip"])
    gallery_dataset = load_gallery_dataset(datasets[dataset_name]["gallery"], datasets[dataset_name]["zip"])

    st.sidebar.markdown("---")
    st.sidebar.subheader("Filter by")
    filter_options = get_filter_options(query_dataset)
    filter_by = st.sidebar.selectbox(
        "Filter by",
        options=filter_options,
        label_visibility="collapsed",
    )
    filter_by = filter_options[filter_by]

    improvement_flag = ""
    if filter_by:
        improved_query_dataset = query_dataset.filter(filter_by, 1)
        worsened_query_dataset = query_dataset.filter(filter_by, -1)
        improvement_flag_options = [
            f"{ImprovementFlags.improvements.value} ({len(improved_query_dataset)})",
            f"{ImprovementFlags.worsenings.value} ({len(worsened_query_dataset)})",
        ]
        improvement_flag = st.sidebar.radio(
            "Filter type",
            options=improvement_flag_options,
            disabled=not bool(filter_by),
        )
        if improvement_flag == improvement_flag_options[0]:
            query_dataset = improved_query_dataset
        else:
            query_dataset = worsened_query_dataset

    if len(query_dataset) == 0:
        st.markdown("There is no query to fulfill the filter requirements.")
        return
    st.sidebar.markdown("---")
    st.sidebar.subheader("Category")
    category_name = st.sidebar.selectbox("Category", query_dataset.categories, label_visibility="collapsed")
    category_dataset = query_dataset.filter("category", category_name)

    st.sidebar.markdown("---")
    more_info_flag = st.sidebar.checkbox(label="Show more info")
    set_session_state(dataset_name, category_name, filter_by, improvement_flag)

    st.title("Query")
    query_viewer = QueryViewer(more_info_flag)
    sample = category_dataset[st.session_state.query_controller_position]
    show_query(query_viewer, sample)

    st.title("Retrieval results")
    top_k = min(query_dataset.max_top_k, TOP_K)
    st.subheader("Baseline model")
    baseline_results_viewer = GalleryViewer(top_k, more_info_flag)
    show_retrieval_results(
        baseline_results_viewer, sample, gallery_dataset, matching_type=RetrievalResultsType.before_stir
    )
    st.subheader("Baseline model + STIR postprocessing")
    stir_results_viewer = GalleryViewer(top_k, more_info_flag)
    show_retrieval_results(stir_results_viewer, sample, gallery_dataset, matching_type=RetrievalResultsType.after_stir)


@st.cache_resource(show_spinner=True)
def download_datasets(datasets: Dict[str, Dict[str, Dict[str, str]]]) -> Dict[str, Dict[str, str]]:
    from pathlib import Path

    import gdown

    output: Dict[str, Dict[str, str]] = defaultdict(dict)
    for dataset_name, dataset_info in datasets.items():
        for data_name, gdrive_id in dataset_info["gdrive_ids"].items():
            local_path = dataset_info["local_paths"][data_name]
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            gdown.download(id=gdrive_id, output=local_path, quiet=False)
            st.success(f"Loaded {data_name} file for {dataset_name} dataset.")
            output[dataset_name][data_name] = dataset_info["local_paths"][data_name]
    return output


def show_query(query_viewer: QueryViewer, sample: QuerySample) -> None:
    image = sample.load_image()
    image = pad_image_to_square(image, SIZE, BORDER_SIZE)
    info: Dict[str, str] = {"Label": str(sample.label), "Category": sample.category}
    query_viewer.show(image, info)


def set_session_state(dataset_name: str, category_name: str, filter_by: str, improvement_flag: str):
    if "query_controller_position" not in st.session_state:
        st.session_state.query_controller_position = 0
    if "dataset_name" not in st.session_state:
        st.session_state.dataset_name = dataset_name
    if "category" not in st.session_state:
        st.session_state.category = category_name
    if "filter_by" not in st.session_state:
        st.session_state.filter_by = filter_by
    if "improvement_flag" not in st.session_state:
        st.session_state.improvement_flag = improvement_flag
    if st.session_state.category != category_name:
        st.session_state.category = category_name
        st.session_state.query_controller_position = 0
    if st.session_state.dataset_name != dataset_name:
        st.session_state.dataset_name = dataset_name
        st.session_state.category = category_name
        st.session_state.query_controller_position = 0
    if st.session_state.filter_by != filter_by:
        st.session_state.filter_by = filter_by
        st.session_state.query_controller_position = 0
    if st.session_state.improvement_flag != improvement_flag:
        st.session_state.improvement_flag = improvement_flag
        st.session_state.query_controller_position = 0


def show_retrieval_results(
    viewer: GalleryViewer, sample: QuerySample, gallery_dataset: GalleryDataset, matching_type: RetrievalResultsType
):
    images = []
    top_k_images_ids = (
        sample.top_k_images_ids
        if matching_type == RetrievalResultsType.before_stir
        else sample.postprocessed_top_k_images_ids
    )
    top_k_scores = (
        sample.top_k_scores if matching_type == RetrievalResultsType.before_stir else sample.postprocessed_top_k_scores
    )
    infos = []
    for image_id, score in zip(top_k_images_ids, top_k_scores):
        gallery_sample = gallery_dataset.find_by_id(image_id)
        image = gallery_sample.load_image()
        if gallery_sample.label == sample.label:
            image = pad_image_to_square(image, SIZE - BORDER_SIZE)
            image = cv2.copyMakeBorder(
                image, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE, cv2.BORDER_CONSTANT, value=GREEN_COLOR
            )
        else:
            image = pad_image_to_square(image, SIZE)
        images.append(image)
        infos.append(
            {
                "Label": gallery_sample.label,
                "Category": gallery_sample.category,
                "Distance": score,
            }
        )
    viewer.show(images, infos)


def get_filter_options(dataset: QueryDataset) -> Dict[str, Union[None, str]]:
    options: Dict[str, Union[None, str]] = {
        c.split("_")[0]: c
        for c in dataset.columns
        if c.endswith(IMPROVED_SUFFIX) and c not in METRICS_TO_EXCLUDE_FROM_VIEWER
    }
    return options


if __name__ == "__main__":
    main()
