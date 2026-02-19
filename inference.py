"""Greedy-decode inference for the Transformer trained on the copy-task."""

import torch
from transformer import Transformer


def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    max_len: int = 50,
    bos_idx: int = 1,
    eos_idx: int = 2,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Greedy auto-regressive decoding.

    Args:
        model:    Trained Transformer.
        src:      Source token ids, shape (1, src_len).
        max_len:  Maximum number of tokens to generate.
        bos_idx:  Beginning-of-sequence token index.
        eos_idx:  End-of-sequence token index.
        device:   Torch device string.

    Returns:
        Tensor of generated token ids, shape (1, generated_len).
    """
    model.eval()
    src = src.to(device)

    # Encode source once
    src_mask = model.make_src_mask(src)
    with torch.no_grad():
        encoder_output = model.encoder(src, src_mask)

    # Start decoder input with BOS token
    tgt = torch.tensor([[bos_idx]], dtype=torch.long, device=device)

    for _ in range(max_len):
        tgt_mask = model.make_tgt_mask(tgt)
        with torch.no_grad():
            decoder_output = model.decoder(tgt, encoder_output, src_mask, tgt_mask)
            logits = model.output_projection(decoder_output)  # (1, seq, vocab)

        # Pick the token with the highest probability at the last position
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (1, 1)
        tgt = torch.cat([tgt, next_token], dim=1)

        if next_token.item() == eos_idx:
            break

    return tgt  # (1, generated_len)


def beam_search(
    model: Transformer,
    src: torch.Tensor,
    beam_size: int = 4,
    max_len: int = 50,
    bos_idx: int = 1,
    eos_idx: int = 2,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Beam-search decoding.

    Returns the highest-scoring complete sequence.
    """
    model.eval()
    src = src.to(device)
    src_mask = model.make_src_mask(src)

    with torch.no_grad():
        encoder_output = model.encoder(src, src_mask)  # (1, src_len, d_model)

    # Expand encoder output for each beam
    encoder_output = encoder_output.expand(beam_size, -1, -1)  # (B, src_len, d)
    src_mask = src_mask.expand(beam_size, -1, -1, -1)

    # Each beam: (sequence, log_prob)
    beams = [(torch.tensor([[bos_idx]], dtype=torch.long, device=device), 0.0)]
    completed = []

    for _ in range(max_len):
        candidates = []
        for seq, score in beams:
            if seq[0, -1].item() == eos_idx:
                completed.append((seq, score))
                continue

            tgt = seq.expand(beam_size, -1)  # rough broadcast for batch decode
            tgt_single = seq  # (1, cur_len)
            tgt_mask = model.make_tgt_mask(tgt_single)

            with torch.no_grad():
                dec_out = model.decoder(
                    tgt_single,
                    encoder_output[:1],  # use first copy of encoder
                    src_mask[:1],
                    tgt_mask,
                )
                logits = model.output_projection(dec_out)  # (1, cur_len, vocab)

            log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)  # (1, vocab)
            top_log_probs, top_ids = log_probs.topk(beam_size, dim=-1)  # (1, B)

            for i in range(beam_size):
                new_token = top_ids[0, i].unsqueeze(0).unsqueeze(0)  # (1,1)
                new_seq = torch.cat([seq, new_token], dim=1)
                new_score = score + top_log_probs[0, i].item()
                candidates.append((new_seq, new_score))

        if not candidates:
            break

        # Keep top beam_size candidates
        candidates.sort(key=lambda x: x[1], reverse=True)
        beams = candidates[:beam_size]

    completed.extend(beams)
    completed.sort(key=lambda x: x[1], reverse=True)
    return completed[0][0]  # best sequence


if __name__ == "__main__":
    VOCAB_SIZE = 100
    device = "cpu"

    model = Transformer(
        src_vocab_size=VOCAB_SIZE,
        tgt_vocab_size=VOCAB_SIZE,
        d_model=128,
        num_layers=2,
        num_heads=4,
        d_ff=256,
    )

    # Load trained weights if available
    try:
        model.load_state_dict(torch.load("best_transformer.pt", map_location=device))
        print("Loaded trained weights from best_transformer.pt")
    except FileNotFoundError:
        print("No trained weights found — using random weights for demo.")

    model.eval()

    # Demo: copy task — source sequence should be reproduced by the decoder
    src_tokens = torch.tensor([[5, 12, 34, 7, 89, 23, 45]], dtype=torch.long)
    print(f"Source:  {src_tokens[0].tolist()}")

    greedy_out = greedy_decode(model, src_tokens, max_len=20, device=device)
    print(f"Greedy:  {greedy_out[0].tolist()}")

    beam_out = beam_search(model, src_tokens, beam_size=4, max_len=20, device=device)
    print(f"Beam(4): {beam_out[0].tolist()}")
