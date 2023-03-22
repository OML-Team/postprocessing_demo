from typing import Dict, List

import cv2
import numpy as np
import streamlit as st
from utils import pad_image_to_square

from data import GalleryDataset, QuerySample
from src.const import BORDER_SIZE, GREEN_COLOR, SIZE, RetrievalResultsType


class QueryViewer:
    def __init__(self, more_info_flag: bool):
        self.more_info_flag = more_info_flag
        tmp_col1 = st.columns([1, 2])[0]
        with tmp_col1:
            self.image_widget = st.columns(1)[0]
            self.prev, self.random, self.next = st.columns([1, 1, 1], gap="small")
        self.prev.button("Prev", on_click=self._add_to_viewer_position, args=(-1,))
        self.random.button("Rand", on_click=self._add_to_viewer_position, args=(np.random.randint(0, 1e10),))
        self.next.button("Next", on_click=self._add_to_viewer_position, args=(1,))

    def _add_to_viewer_position(self, v: int):
        st.session_state.query_controller_position += v

    def show(self, image: np.ndarray, info: Dict[str, str]):
        show_image_card(self.image_widget, image, info, self.more_info_flag)


class GalleryViewer:
    def __init__(self, n: int, more_info_flag: bool):
        self.more_info_flag = more_info_flag
        self.cols = st.columns(n)

    def show(self, images: List[np.ndarray], infos: List[Dict[str, str]]):
        for col, image, info in zip(self.cols, images, infos):
            show_image_card(col, image, info, self.more_info_flag)


def show_image_card(st_, image, info, show_info_flag: bool) -> None:
    st_.image(image)
    if show_info_flag:
        for k, v in info.items():
            st_.markdown(f"**{k}**: {v}")


def show_query(query_viewer: QueryViewer, sample: QuerySample) -> None:
    image = sample.load_image()
    image = pad_image_to_square(image, SIZE, BORDER_SIZE)
    info: Dict[str, str] = {"Label": str(sample.label), "Category": sample.category}
    query_viewer.show(image, info)


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
