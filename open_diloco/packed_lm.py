"""
문서 단위 스트림을 seq_length로 패킹하고, 문서 경계에서 attention을 끊기 위한
position_ids를 제공합니다.

- 샤딩: 패킹 전에 split_dataset_* / hash_assign이 적용된 Iterable을 받습니다.
- 학습: 새 transformers(>=4.48)는 position_ids가 문서마다 0부터 시작하면
  create_causal_mask → find_packed_sequence_indices에서 자동으로 block-causal 마스크를 만듭니다.
  이때 attention_mask는 None(또는 넘기지 않음)이어야 감지가 작동합니다.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Dict, Iterator, List

import torch
from torch.utils.data import IterableDataset


def intra_document_position_ids(segment_ids: torch.Tensor) -> torch.Tensor:
    """segment_ids (B, L) → position_ids (B, L): 문서가 바뀔 때마다 0부터 다시 셉니다.
    cummax 기반 벡터 연산으로 Python for-loop 없이 처리합니다."""
    B, L = segment_ids.shape
    device = segment_ids.device
    boundary = torch.zeros(B, L, dtype=torch.bool, device=device)
    boundary[:, 0] = True
    if L > 1:
        boundary[:, 1:] = segment_ids[:, 1:] != segment_ids[:, :-1]
    indices = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
    marks = torch.where(boundary, indices, torch.tensor(-1, dtype=torch.long, device=device))
    last_boundary, _ = marks.cummax(dim=1)
    return indices - last_boundary


def collate_packed_causal_lm(features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """패킹된 샘플 리스트를 배치 텐서로 묶습니다.
    attention_mask를 넘기지 않아야 transformers의 packed sequence 자동 감지가 작동합니다.
    """
    input_ids = torch.stack([torch.tensor(f["input_ids"], dtype=torch.long) for f in features], dim=0)
    position_ids = torch.stack([torch.tensor(f["position_ids"], dtype=torch.long) for f in features], dim=0)
    labels = input_ids.clone()
    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "labels": labels,
    }


class PackedSequenceIterable(IterableDataset):
    """
    토큰화된 IterableDataset(행마다 가변 길이 input_ids)을 읽어 EOS로 문서를 구분하며 seq_length로 자릅니다.
    """

    def __init__(
        self,
        source,
        eos_token_id: int,
        seq_length: int,
    ):
        super().__init__()
        self._source = source
        self.eos_token_id = eos_token_id
        self.seq_length = seq_length

    def __iter__(self) -> Iterator[Dict[str, List[int]]]:
        buf = deque()
        seg_buf = deque()
        doc_id = 0
        sl = self.seq_length
        eos = self.eos_token_id

        for ex in self._source:
            ids = ex.get("input_ids")
            if not ids:
                continue
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            elif not isinstance(ids, list):
                ids = list(ids)

            n = len(ids)
            buf.extend(ids)
            seg_buf.extend([doc_id] * n)
            buf.append(eos)
            seg_buf.append(doc_id)
            doc_id += 1

            while len(buf) >= sl:
                chunk_tok = [buf.popleft() for _ in range(sl)]
                chunk_seg = [seg_buf.popleft() for _ in range(sl)]

                seg_t = torch.tensor(chunk_seg, dtype=torch.long)
                pos_t = intra_document_position_ids(seg_t.unsqueeze(0)).squeeze(0)
                yield {
                    "input_ids": chunk_tok,
                    "position_ids": pos_t.tolist(),
                }
