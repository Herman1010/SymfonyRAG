def expand_with_neighbors(hit, all_metas, window: int = 1, suffix: str = "fixed"):
    """
    Expand text by taking neighbors from the same source:
    chunk_id format expected: {source}_{suffix}_{idx}  e.g. routing.rst_fixed_19
    """
    source = hit.get("source")
    chunk_id = hit.get("chunk_id", "")

    # extract idx
    try:
        idx = int(chunk_id.split("_")[-1])
    except:
        return hit

    if not source:
        return hit

    # build a map for fast access
    by_id = {m.get("chunk_id"): m for m in all_metas}

    neighbors = []
    for delta in range(-window, window + 1):
        target_id = f"{source}_{suffix}_{idx + delta}"
        m = by_id.get(target_id)
        if m and m.get("text"):
            neighbors.append(m["text"])

    if not neighbors:
        return hit

    new_hit = dict(hit)
    new_hit["text"] = "\n\n".join(neighbors)
    new_hit["expanded_from"] = chunk_id
    new_hit["window"] = window
    return new_hit
