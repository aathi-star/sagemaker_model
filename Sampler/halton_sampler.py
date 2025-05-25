import torch
import random
import math
import numpy as np


class HaltonSampler(object):
    """
    HaltonSampler provides a deterministic (or once-shuffled) schedule for revealing tokens.
    It's designed to be used with a controlling sampler loop like in MaskGIT.sample.
    """

    def __init__(self, input_size: int, num_steps: int, sched_pow: float = 1.0, randomize: bool = False, top_k_default: int = -1):
        """
        Initializes the HaltonSampler.

        Args:
            input_size (int): The side length of the square token map (e.g., 16 for 16x16 tokens).
            num_steps (int): Total number of sampling iteration steps.
            sched_pow (float): Exponent for the cosine schedule power.
            randomize (bool): If True, the Halton sequence is shuffled once for the lifetime of this sampler instance.
            top_k_default (int): Default top-k value for sampling if not overridden in sample_step.
        """
        super().__init__()
        self.input_size = input_size
        self.num_steps = num_steps
        self.sched_pow = sched_pow # Used by _calculate_num_known_tokens
        self.randomize = randomize
        self.top_k_default = top_k_default
        self.K = self.input_size ** 2

        # Build and potentially shuffle the Halton sequence indices
        base_halton_indices = self.build_halton_mask(self.input_size) # Shape (K, 2)
        if self.randomize:
            roll_amount = torch.randint(0, self.K, (1,)).item()
            self.active_halton_indices = torch.roll(base_halton_indices.clone(), roll_amount, 0)
        else:
            self.active_halton_indices = base_halton_indices

    def __str__(self):
        return f"HaltonSampler(steps={self.num_steps}, sched_pow={self.sched_pow}, randomize={self.randomize}, top_k_default={self.top_k_default})"

    def _calculate_num_known_tokens(self, step_idx: int) -> int:
        """Calculates how many tokens should be known by the end of step_idx."""
        if step_idx < 0:
            return 0
        
        ratio = (step_idx + 1) / self.num_steps
        # Cosine schedule for the ratio of known tokens
        # Clamp ratio to avoid domain errors with acos, especially at ratio=1.0 for the last step.
        clamped_ratio = torch.clamp(torch.tensor(ratio, dtype=torch.float32), 0.0, 1.0)
        
        # The pi/2 factor normalizes acos range [0, pi/2] for ratio in [1,0] to [0,1]
        # So, (1 - acos(x) / (pi/2)) goes from 0 to 1 as x goes from 0 to 1.
        if clamped_ratio.item() == 1.0:
            norm_known_ratio = torch.tensor(1.0, dtype=torch.float32)
        else:
            norm_known_ratio = (1.0 - (torch.acos(clamped_ratio) / (math.pi / 2.0)))
        
        # Apply schedule power to the normalized ratio
        powered_norm_known_ratio = torch.pow(norm_known_ratio, self.sched_pow)
        
        num_known = int(powered_norm_known_ratio.item() * self.K)
        # Ensure the number of known tokens is within bounds [0, K]
        num_known = min(self.K, max(0, num_known))
        return num_known

    def get_schedule_func(self, power: float = None, K: int = None):
        """
        Returns a function that, given a step index, provides a boolean mask 
        of tokens to be predicted at that step.
        `power` and `K` args are for compatibility, but internal values are preferred.
        """
        # Use instance's sched_pow if not overridden
        # current_sched_power = power if power is not None else self.sched_pow
        # K is self.input_size ** 2, already stored as self.K

        def schedule_fn(current_step_idx: int, device: torch.device) -> torch.Tensor:
            """
            Args:
                current_step_idx (int): The current sampling step, 0-indexed.
                device (torch.device): The device for the output mask tensor.
            Returns:
                torch.Tensor: A boolean mask of shape (input_size, input_size) 
                              where True indicates tokens to be predicted at this step.
            """
            num_known_at_prev_step = self._calculate_num_known_tokens(current_step_idx - 1)
            num_known_at_curr_step = self._calculate_num_known_tokens(current_step_idx)

            # Ensure num_known_at_curr_step is at least one more than prev_step, up to K
            # This guarantees progress, especially with integer rounding.
            # And also ensures at least one token is scheduled if num_known_at_curr_step is same as prev due to rounding.
            if num_known_at_curr_step == num_known_at_prev_step and num_known_at_curr_step < self.K:
                 num_known_at_curr_step = min(self.K, num_known_at_prev_step +1)
            
            # Indices of tokens to be predicted in this step
            # These are from the pre-shuffled active_halton_indices
            token_indices_for_this_step = self.active_halton_indices[num_known_at_prev_step:num_known_at_curr_step]

            # Create the boolean mask
            bool_mask = torch.zeros(self.input_size, self.input_size, dtype=torch.bool, device=device)
            if token_indices_for_this_step.numel() > 0:
                rows = token_indices_for_this_step[:, 0].long() # Ensure rows are long type
                cols = token_indices_for_this_step[:, 1].long() # Ensure cols are long type
                bool_mask[rows, cols] = True
            
            return bool_mask

        return schedule_fn

    def sample_step(self, logits: torch.Tensor, current_step_idx: int, total_steps: int, temperature: float, top_k_val: int):
        """
        Samples tokens based on the logits provided for the current step.

        Args:
            logits (torch.Tensor): Raw output logits from the model (already CFG-processed)
                                   for the tokens to be sampled in this step.
                                   Shape: (num_tokens_to_sample, vocab_size)
            current_step_idx (int): Current iteration step (0 to total_steps-1).
            total_steps (int): Total number of sampling steps.
            temperature (float): Temperature for softmax scaling.
            top_k_val (int): If > 0, performs top-k sampling. Otherwise, standard categorical sampling.

        Returns:
            torch.Tensor: Predicted token indices. Shape: (num_tokens_to_sample,)
        """
        # Apply temperature
        # Ensure temperature is not zero to avoid division by zero
        effective_temp = max(temperature, 1e-6) 
        probs = torch.softmax(logits / effective_temp, dim=-1) # Shape: (num_tokens_to_sample, vocab_size)

        num_tokens_to_sample, vocab_size = logits.shape

        # Use top_k_val if provided and positive, otherwise self.top_k_default
        current_top_k = top_k_val if top_k_val > 0 else self.top_k_default

        if current_top_k > 0:
            # Ensure K is not larger than vocab_size
            actual_k = min(current_top_k, vocab_size)
            
            top_k_probs, top_k_indices = torch.topk(probs, actual_k, dim=-1) # Shape: (num_tokens_to_sample, actual_k)
            
            # Normalize the top-k probabilities to sum to 1 for multinomial sampling
            norm_top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-9) # Shape: (num_tokens_to_sample, actual_k)
            
            # Sample indices from the top_k probabilities
            # multinomial expects input (N, K)
            sampled_indices_in_top_k = torch.multinomial(norm_top_k_probs, num_samples=1) # Shape: (num_tokens_to_sample, 1)
            
            # Gather the actual token indices corresponding to the sampled top-k positions
            # top_k_indices is (num_tokens_to_sample, actual_k)
            # sampled_indices_in_top_k is (num_tokens_to_sample, 1)
            pred_tokens = torch.gather(top_k_indices, -1, sampled_indices_in_top_k)
            pred_tokens = pred_tokens.squeeze(-1) # Shape: (num_tokens_to_sample,)
            
        else:
            # Standard categorical sampling from the full probability distribution
            # probs has shape (num_tokens_to_sample, vocab_size)
            pred_tokens = torch.distributions.Categorical(probs=probs).sample() # Shape: (num_tokens_to_sample,)

        return pred_tokens

    def build_halton_mask(self, input_size, nb_point=10_000):
        """ Generate a halton 'quasi-random' sequence in 2D.
          :param
            input_size -> int: size of the mask, (input_size x input_size).
            nb_point   -> int: number of points to be sample, it should be high to cover the full space.
            h_base     -> torch.LongTensor: seed for the sampling.
          :return:
            mask -> Torch.LongTensor: (input_size x input_size) the mask where each value corresponds to the order of sampling.
        """

        def halton(b, n_sample):
            """Naive Generator function for Halton sequence."""
            n, d = 0, 1
            res = []
            for index in range(n_sample):
                x = d - n
                if x == 1:
                    n = 1
                    d *= b
                else:
                    y = d // b
                    while x <= y:
                        y //= b
                    n = (b + 1) * y - x
                res.append(n / d)
            return res

        # Sample 2D mask
        data_x = torch.tensor(halton(2, nb_point)).view(-1, 1)
        data_y = torch.tensor(halton(3, nb_point)).view(-1, 1)
        mask = torch.cat([data_x, data_y], dim=1) * input_size
        mask = torch.floor(mask)

        # remove duplicate
        indexes = np.unique(mask.numpy(), return_index=True, axis=0)[1]
        mask = [mask[index].numpy().tolist() for index in sorted(indexes)]
        return torch.tensor(np.array(mask))

    def __call__(self, trainer, init_code=None, nb_sample=50, labels=None, verbose=True):
        """
        Runs the Halton-based sampling process. (OLD IMPLEMENTATION - for reference)

        Args:
            trainer    -> MaskGIT: The model trainer.
            init_code  -> torch.Tensor: Pre-initialized latent code.
            nb_sample  -> int: Number of images to generate.
            labels     -> torch.Tensor: Class labels for conditional generation.
            verbose    -> bool: Whether to display progress.

        Returns:
            Tuple: Generated images, list of intermediate codes, list of masks used during generation.
        """
        raise NotImplementedError("This __call__ method is deprecated. Use MaskGIT.sample with this sampler instance.")
