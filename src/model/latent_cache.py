from typing import List, Tuple

import torch


class LatentCache:
    def __init__(self) -> None:
        """list for layers"""
        self.latent_cache: List[torch.Tensor] = []

    def has_item(self, layer_idx) -> bool:
        return len(self.latent_cache) > layer_idx

    def num_items(self) -> int:
        if len(self.latent_cache) == 0:
            return 0
        else:
            # The shape of the key_cache is [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
            return self.key_cache[0].shape[-2]

    def get(self, layer_idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.latent_cache[layer_idx]

    def update(
        self,
        latent_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.latent_states) <= layer_idx:
            # If we never added anything to the KV-Cache of this layer, let's create it.
            self.latent_cache.append(latent_states)
        else:
            # ... otherwise we concatenate the new keys with the existing ones.
            # each tensor has shape: [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
            self.latent_cache[layer_idx] = torch.cat(
                [self.latent_cache[layer_idx], latent_states], dim=-2
            )

        # ... and then we return all the existing keys + the new ones.
        return self.latent_cache[layer_idx]