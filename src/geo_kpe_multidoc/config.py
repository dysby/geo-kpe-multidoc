from dataclasses import dataclass
from enum import Enum

CandidateMode = Enum(
    "CandidateMode",
    ["AvgContext", "global_attention", "MaskAll", "MaskFirst", "MaskHighest", "other"],
)
AttentionMode = Enum("AttentionMode", "", "global_attention")
Similarity = Enum("Similarity", ["cosine_similarity", "MMR"])
PostProcessing = Enum("PostProcessing", ["z_score"])


@dataclass
class ExperimentConfig:
    """
    Candidate selection strategy options configuration
    """

    candidate_mode: CandidateMode
    attention: AttentionMode
    similarity: Similarity
    diversity: str
    post_processing: str
    cache_pos_tags: bool  # replace pos_tag_memory
