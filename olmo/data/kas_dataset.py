from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from olmo.exceptions import OLMoEnvironmentError

from ..aliases import PathOrStr
from ..config import InstanceFilterConfig
from ..util import _get_s3_client, file_size, get_bytes_range
from .util import find_periodic_sequences, get_document_lengths

from .memmap_dataset import MemMapDataset
import os
import math

__all__ = ["KASDataset"]


def map_data_to_metadata(data_path: str) -> str:
    base, ext = os.path.splitext(data_path)
    return f"{base}.csv.gz"

class KASDataset(MemMapDataset):
    def __init__(self, *args, **kwargs):
        self.sentence_boundaries = kwargs.pop('sentence_boundaries', None)
        paths = kwargs.pop('paths', [])
        super().__init__(*paths, **kwargs)
        # Load and sort metadata for each data file
        metadata_list = [
            (path, self.load_metadata(map_data_to_metadata(path)))
            for path in paths
        ]
        # Sort by the start index of the first row in each metadata
        metadata_list.sort(key=lambda row: row[1][0]['start'])
        # Convert to an ordered dictionary
        self._metadata = {}
        for i, (path, metadata) in enumerate(metadata_list):
            self._metadata[i] = {}
            for j, row in enumerate(metadata):
                self._metadata[i][j] = row

    def load_metadata(self, metadata_path: str) -> Dict[str, Any]:
        import pandas as pd
        column_names = ['start', 'end', 'id', 'src', 'loc', 'title', 'entities']
        df = pd.read_csv(metadata_path, names=column_names)
        return df.to_dict(orient='records')

    def _read_chunk_from_memmap(self, path: PathOrStr, memmap_index: int, index: int, dtype=None) -> torch.Tensor:
        dtype = dtype or self.dtype
        item_size = dtype(0).itemsize
        metadata = self._metadata[memmap_index][index]
        chunk_size = metadata["end"] - metadata["start"]
        bytes_start = item_size * metadata["start"]
        num_bytes = item_size * chunk_size
        buffer = get_bytes_range(path, bytes_start, num_bytes)
        array = np.frombuffer(buffer, dtype=dtype)
        if dtype == np.bool_:
            return torch.tensor(array)
        else:
            return torch.tensor(array.astype(np.int_), dtype=torch.long)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        index = int(index)  # in case this is a numpy int type.
        pos_index = index if index >= 0 else len(self) + index

        # The index of the memmap array within 'self.memmaps'
        memmap_index: Optional[int] = None
        # The 'index' relative to the corresponding memmap array.
        memmap_local_index: Optional[int] = None
        for i, (offset_start, offset_end) in enumerate(self.offsets):
            if offset_start <= pos_index < offset_end:
                memmap_index = i
                memmap_local_index = pos_index - offset_start

        if memmap_index is None or memmap_local_index is None:
            raise IndexError(f"{index} is out of bounds for dataset of size {len(self)}")

        # Read the data from file.
        input_ids = self._read_chunk_from_memmap(self._memmap_paths[memmap_index], memmap_index, memmap_local_index)
        out: Dict[str, Any] = {"input_ids": input_ids}
        if self.instance_filter_config is not None:
            out["instance_mask"] = self._validate_instance(input_ids)

        if self._label_mask_paths is not None:
            label_mask = self._read_chunk_from_memmap(
                self._label_mask_paths[memmap_index], memmap_index, memmap_local_index, dtype=np.bool_
            )
            out["label_mask"] = label_mask

        if self._include_instance_metadata:
            metadata = self._metadata[memmap_index][index]
            out["metadata"] = deepcopy(metadata)

        if self._generate_attention_mask:
            assert self._pad_token_id is not None
            attn_mask = torch.ones_like(input_ids)
            attn_mask.masked_fill_(input_ids == self._pad_token_id, 0)
            out["attention_mask"] = attn_mask

        if self._generate_doc_lengths:
            assert self._eos_token_id is not None
            out["doc_lens"] = get_document_lengths(input_ids, self._eos_token_id)

        return out
