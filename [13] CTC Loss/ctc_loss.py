import torch

def ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0, reduction="none"):
    seq_len, batch_size, _ = log_probs.shape
    device = log_probs.device
    neg_inf = torch.finfo(log_probs.dtype).min
    B = torch.arange(batch_size, device=device)

    # Prepare targets with interleaved blanks
    _targets = torch.cat([targets, torch.zeros(batch_size, 1, device=device, dtype=torch.long)], dim=-1)
    _targets = torch.stack([torch.full_like(_targets, blank), _targets], dim=-1).flatten(start_dim=-2)

    # Identification of labels that are different from the one two positions prior
    diff_labels = torch.cat([
        torch.tensor([[False, False]], device=device).expand(batch_size, -1),
        _targets[:, 2:] != _targets[:, :-2]
    ], dim=-1)

    # Gather log probabilities corresponding to the target sequence
    log_probs_targets = log_probs.gather(dim=-1, index=_targets.expand(seq_len, -1, -1))

    # Initialize DP table (log_alpha)
    # Shape: (T, B, S + 2) to simplify indexing for transitions
    log_alpha = torch.full((seq_len, batch_size, _targets.shape[-1] + 2), neg_inf, device=device)
    
    # Initial state
    log_alpha[0, :, 2] = log_probs[0, :, blank]
    log_alpha[0, :, 3] = log_probs[0, B, _targets[:, 1]]

    # Dynamic Programming loop
    for t in range(1, seq_len):
        prev_stay = log_alpha[t-1, :, 2:]
        prev_next = log_alpha[t-1, :, 1:-1]
        prev_skip = torch.where(diff_labels, log_alpha[t-1, :, :-2], neg_inf)

        combined = torch.stack([prev_stay, prev_next, prev_skip])
        log_alpha[t, :, 2:] = log_probs_targets[t] + torch.logsumexp(combined, dim=0)

    # Extract final log probabilities at specific input lengths
    final_log_alpha = log_alpha[input_lengths - 1, B]

    # Map to the two valid ending positions (last label or final blank)
    ending_label_idx = 2 + target_lengths * 2 - 1
    ending_blank_idx = 2 + target_lengths * 2
    
    indices = torch.stack([ending_label_idx, ending_blank_idx], dim=-1)
    final_probs = final_log_alpha.gather(dim=-1, index=indices)

    loss = -torch.logsumexp(final_probs, dim=-1)

    if reduction == "sum":
        return torch.sum(loss)
    elif reduction == "mean":
        return torch.mean(loss)
    return loss