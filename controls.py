from typing import Dict, List

import numpy as np
import streamlit as st


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
