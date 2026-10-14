"""GazePrune: per-token depth pruning of visual tokens in VLA models, driven by where the action queries look.

A small ranker reads the hidden states at the input of layer 2 and orders the visual tokens; a fixed per-layer
schedule (from the oracle depth DP on teacher attention labels) decides how many of them each layer keeps.
"""

from gazeprune.ranker import DepthRanker, combine_heads, depth_from_scores, query_features
from gazeprune.predictor import TokenDepthPredictor
from gazeprune.schedule import oracle_depth_dp, round_schedule

__all__ = ["DepthRanker", "combine_heads", "depth_from_scores", "query_features", "TokenDepthPredictor",
           "oracle_depth_dp", "round_schedule"]
