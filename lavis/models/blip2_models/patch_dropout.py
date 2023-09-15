import torch

class PatchDropout(torch.nn.Module):
    """ 
    Implementation modified from: https://github.com/yueliukth/PatchDropout/blob/main/scripts/patchdropout.py
    Implements PatchDropout: https://arxiv.org/abs/2208.07220
    Adds capability to sample tokens from tubelets (i.e. identical spatial location in consecutive frames)
    in addition to the regular sampling from single frames.
    """
    def __init__(self, p=0.0, sampling="tubelet_uniform", token_shuffling=False, tokens_per_frame=196, num_frames=4):
        super().__init__()
        assert 0 <= p < 1, "The dropout rate p must be in [0,1)"
        self.tokens_per_frame = tokens_per_frame
        self.keep_rate = 1 - p
        self.sampling = sampling
        self.token_shuffling = token_shuffling
        self.num_frames = num_frames
        self.n_keep = int(self.tokens_per_frame * (1 - p)) # number of frames to keep per patch (if tubelet sampling is used)

    def forward(self, x, force_drop=False):
        """
        If force drop is true it will drop the tokens also during inference.
        """
        if not self.training and not force_drop: return x        
        if self.keep_rate == 1: return x

        # batch, length, dim
        N, L, D = x.shape
        
        # making cls mask (assumes that CLS is always the 1st element)
        cls_mask = torch.zeros(N, 1, dtype=torch.int64, device=x.device)
        # generating patch mask
        patch_mask = self.get_mask(x)

        # cat cls and patch mask
        patch_mask = torch.hstack([cls_mask, patch_mask])
        # gather tokens
        x = torch.gather(x, dim=1, index=patch_mask.unsqueeze(-1).repeat(1, 1, D))

        return x
    
    def get_mask(self, x):
        if self.sampling == "uniform":
            return self.uniform_mask(x)
        elif self.sampling == "tubelet_uniform":
            return self.tubelet_uniform_mask(x)
        else:
            raise NotImplementedError(f"PatchDropout does not support {self.sampling} sampling")
            return None

    def uniform_mask(self, x):
        """
        Returns an id-mask using uniform sampling
        """
        N, L, D = x.shape
        _L = L -1 # patch lenght (without CLS)
        
        keep = self.n_keep
        patch_mask = torch.rand(N, _L, device=x.device)
        patch_mask = torch.argsort(patch_mask, dim=1) + 1
        patch_mask = patch_mask[:, :keep]
        if not self.token_shuffling:
            patch_mask = patch_mask.sort(1)[0]
        return patch_mask
    
    def tubelet_uniform_mask(self, x):
        """
        Returns an id-mask using uniform sampling
        """
        N, L, D = x.shape
        if L == self.tokens_per_frame + 1: # Image input
            return self.uniform_mask(x)
        _L = self.tokens_per_frame # patch length (without CLS)
        keep = self.n_keep
        #import pdb; pdb.set_trace()
        patch_mask = torch.rand(N, _L, device=x.device)
        patch_mask = torch.argsort(patch_mask, dim=1) 
        patch_mask = patch_mask[:, :keep]
        # Repeat the same mask for all frames for mask tubelets
        repeated_patch_mask = patch_mask.repeat(1, self.num_frames)
        values_to_add = self.tokens_per_frame * torch.arange(0, self.num_frames).repeat_interleave(keep).to(x.device)

        patch_mask = repeated_patch_mask + values_to_add 
        patch_mask = patch_mask + 1 # add 1 to account for CLS token (assumes it is leading token)
        
        if not self.token_shuffling:
            patch_mask = patch_mask.sort(1)[0]
        else:
            raise NotImplementedError("Token shuffling is not implemented for tubelet_uniform_mask")
        return patch_mask


# # Test PatchDropout
if __name__ == "__main__":
    batch_size = 1
    num_frames = 4
    tokens_per_frame = 9
    n_channels = 1
    dropout = PatchDropout(p=0.5, sampling="uniform_tubelet", tokens_per_frame=tokens_per_frame, num_frames=num_frames)
    total_num = batch_size * num_frames * tokens_per_frame * n_channels
    inp = torch.arange(total_num + 1) # add one for cls token
    inp = inp.reshape(batch_size, 1 + num_frames * tokens_per_frame, n_channels)
    print(inp.shape)
    out = dropout(inp)
    print(out.shape)