import zipfile
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, List
from zipfile import ZipFile

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from const import (
    CATEGORIES_COLUMN,
    ID_COLUMN,
    LABELS_COLUMN,
    PATHS_COLUMN,
    POSTPROCESSED_TOP_K_IMAGE_ID_COLUMN_TEMPLATE,
    POSTPROCESSED_TOP_K_SCORE_COLUMN_TEMPLATE,
    TOP_K_IMAGE_ID_COLUMN_TEMPLATE,
    TOP_K_SCORE_COLUMN_TEMPLATE,
    PathType,
)


@dataclass
class Sample:
    id: int
    path: PathType
    category: str
    label: int
    zip: ZipFile

    def load_image(self):
        return np.array(Image.open(self.zip.open(self.path)))


@dataclass
class QuerySample(Sample):
    top_k_scores: List[float]
    top_k_images_ids: List[int]
    postprocessed_top_k_scores: List[float]
    postprocessed_top_k_images_ids: List[int]


class GallerySample(Sample):
    pass


class Dataset:
    def __init__(self, csv_path: PathType, zip_path: PathType):
        self.data = pd.read_csv(csv_path)
        self.zip_path = zip_path

    @property
    def columns(self) -> List[str]:
        return self.data.columns

    @property
    def categories(self):
        return sorted(set(self.data[CATEGORIES_COLUMN]))

    def filter(self, property_name: str, property_value: Any) -> "Dataset":
        dataset = deepcopy(self)
        dataset.data = dataset.data[dataset.data[property_name] == property_value]
        return dataset

    def find_by_id(self, image_id: int):
        item = self.data.index[self.data[ID_COLUMN] == image_id][0]
        return self[item]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item) -> Sample:
        raise NotImplementedError()


class QueryDataset(Dataset):
    @property
    def max_top_k(self):
        return max(int(s.split("_")[1]) for s in self.data.columns if s.startswith("top_"))

    def __getitem__(self, item) -> QuerySample:
        item = item % len(self)

        row = self.data.iloc[item]
        top_k_scores = [self.data.iloc[item][TOP_K_SCORE_COLUMN_TEMPLATE % i] for i in range(1, self.max_top_k + 1)]
        top_k_images_ids = [
            self.data.iloc[item][TOP_K_IMAGE_ID_COLUMN_TEMPLATE % i] for i in range(1, self.max_top_k + 1)
        ]
        postprocessed_top_k_scores = [
            self.data.iloc[item][POSTPROCESSED_TOP_K_SCORE_COLUMN_TEMPLATE % i] for i in range(1, self.max_top_k + 1)
        ]
        postprocessed_top_k_images_ids = [
            self.data.iloc[item][POSTPROCESSED_TOP_K_IMAGE_ID_COLUMN_TEMPLATE % i] for i in range(1, self.max_top_k + 1)
        ]
        return QuerySample(
            id=row[ID_COLUMN],
            path=row[PATHS_COLUMN],
            category=row[CATEGORIES_COLUMN],
            label=row[LABELS_COLUMN],
            zip=zipfile.ZipFile(self.zip_path),
            top_k_scores=top_k_scores,
            top_k_images_ids=top_k_images_ids,
            postprocessed_top_k_scores=postprocessed_top_k_scores,
            postprocessed_top_k_images_ids=postprocessed_top_k_images_ids,
        )


class GalleryDataset(Dataset):
    def __getitem__(self, item) -> GallerySample:
        item = item % len(self)

        row = self.data.iloc[item]
        return GallerySample(
            id=row[ID_COLUMN],
            path=row[PATHS_COLUMN],
            category=row[CATEGORIES_COLUMN],
            label=row[LABELS_COLUMN],
            zip=zipfile.ZipFile(self.zip_path),
        )


@st.cache_data
def load_query_dataset(dataset_path: PathType, zip_path: PathType):
    return QueryDataset(dataset_path, zip_path)


@st.cache_data
def load_gallery_dataset(dataset_path: PathType, zip_path: PathType):
    return GalleryDataset(dataset_path, zip_path)
