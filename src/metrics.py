def compute_hit_at_k(all_hits: list, k: int) -> bool:
    
    hit = False
    for val in all_hits:
        if val <= k:
            hit = True
            break

    return hit
