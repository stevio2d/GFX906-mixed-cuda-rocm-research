#!/usr/bin/env python3

import argparse
import json
import statistics
from pathlib import Path


def median(values):
    return statistics.median(values) if values else 0.0


def top_share(entries, total):
    if not total:
        return 0.0, 0.0
    top1 = entries[0]["count"] / total if entries else 0.0
    topn = sum(entry["count"] for entry in entries) / total
    return top1, topn


def format_float(value):
    return f"{value:.4f}".rstrip("0").rstrip(".")


def layer_shard_imbalance(nodes_by_layer):
    result = []
    for layer, nodes in sorted(nodes_by_layer.items()):
        shard_selections = []
        shard_unique = []
        for node in nodes:
            if not shard_selections:
                shard_selections = [0] * len(node.get("shard_selections", []))
                shard_unique = [0] * len(node.get("shard_unique", []))
            for idx, value in enumerate(node.get("shard_selections", [])):
                shard_selections[idx] += value
            for idx, value in enumerate(node.get("shard_unique", [])):
                shard_unique[idx] += value
        if not shard_selections:
            continue
        avg = sum(shard_selections) / len(shard_selections)
        imbalance = (max(shard_selections) / avg) if avg else 0.0
        avg_unique = sum(node["unique_experts"] for node in nodes) / len(nodes)
        result.append({
            "layer": layer,
            "imbalance": imbalance,
            "shard_selections": shard_selections,
            "shard_unique": shard_unique,
            "avg_node_unique": avg_unique,
        })
    return result


def summarize_profile(path: Path, show_layers: int):
    data = json.loads(path.read_text())
    layers = data.get("layers", [])
    nodes = data.get("nodes", [])
    nodes_by_layer = {}
    for node in nodes:
        nodes_by_layer.setdefault(node["layer"], []).append(node)

    unique_avg = [layer["unique_experts_avg"] for layer in layers]
    max_unique = [layer["max_unique_experts"] for layer in layers]
    top1_shares = []
    top8_shares = []
    for layer in layers:
        top1, top8 = top_share(layer.get("top_experts", []), layer["expert_selections"])
        top1_shares.append(top1)
        top8_shares.append(top8)

    lines = []
    lines.append(f"FILE {path}")
    lines.append(
        f"  layers={len(layers)} nodes={len(nodes)} "
        f"unique_avg_range={min(unique_avg)}-{max(unique_avg)} "
        f"unique_avg_median={format_float(median(unique_avg))} "
        f"max_unique_range={min(max_unique)}-{max(max_unique)} "
        f"max_unique_median={format_float(median(max_unique))}"
    )
    lines.append(
        f"  top1_share_range={format_float(min(top1_shares))}-{format_float(max(top1_shares))} "
        f"top1_share_median={format_float(median(top1_shares))} "
        f"top8_share_range={format_float(min(top8_shares))}-{format_float(max(top8_shares))} "
        f"top8_share_median={format_float(median(top8_shares))}"
    )

    imbalances = layer_shard_imbalance(nodes_by_layer)
    if imbalances:
        lines.append(
            f"  layer_shard_imbalance_max={format_float(max(item['imbalance'] for item in imbalances))} "
            f"layer_shard_imbalance_median={format_float(median([item['imbalance'] for item in imbalances]))}"
        )
        lines.append("  most_imbalanced_layers:")
        for item in sorted(imbalances, key=lambda item: item["imbalance"], reverse=True)[:show_layers]:
            lines.append(
                f"    layer={item['layer']} imbalance={format_float(item['imbalance'])} "
                f"avg_node_unique={format_float(item['avg_node_unique'])} "
                f"selections={item['shard_selections']} unique={item['shard_unique']}"
            )

    lines.append("  narrowest_layers:")
    for layer in sorted(layers, key=lambda item: item["unique_experts_avg"])[:show_layers]:
        top1, top8 = top_share(layer.get("top_experts", []), layer["expert_selections"])
        lines.append(
            f"    layer={layer['layer']} unique_avg={format_float(layer['unique_experts_avg'])} "
            f"max_unique={layer['max_unique_experts']} "
            f"top1={format_float(top1)} top8={format_float(top8)}"
        )

    lines.append("  widest_layers:")
    for layer in sorted(layers, key=lambda item: item["unique_experts_avg"], reverse=True)[:show_layers]:
        top1, top8 = top_share(layer.get("top_experts", []), layer["expert_selections"])
        lines.append(
            f"    layer={layer['layer']} unique_avg={format_float(layer['unique_experts_avg'])} "
            f"max_unique={layer['max_unique_experts']} "
            f"top1={format_float(top1)} top8={format_float(top8)}"
        )

    return data, lines


def compare_profiles(lhs_path: Path, lhs_data, rhs_path: Path, rhs_data, show_layers: int):
    lhs_layers = {layer["layer"]: layer for layer in lhs_data.get("layers", [])}
    rhs_layers = {layer["layer"]: layer for layer in rhs_data.get("layers", [])}
    overlaps = []
    for layer in sorted(set(lhs_layers) & set(rhs_layers)):
        lhs_top = {entry["expert_id"] for entry in lhs_layers[layer].get("top_experts", [])}
        rhs_top = {entry["expert_id"] for entry in rhs_layers[layer].get("top_experts", [])}
        union = lhs_top | rhs_top
        score = (len(lhs_top & rhs_top) / len(union)) if union else 1.0
        overlaps.append({
            "layer": layer,
            "score": score,
            "overlap": sorted(lhs_top & rhs_top),
        })

    lines = []
    lines.append(f"COMPARE {lhs_path.name} vs {rhs_path.name}")
    if overlaps:
        scores = [item["score"] for item in overlaps]
        lines.append(
            f"  top8_overlap_jaccard_min={format_float(min(scores))} "
            f"median={format_float(median(scores))} max={format_float(max(scores))}"
        )
        lines.append("  lowest_overlap_layers:")
        for item in sorted(overlaps, key=lambda item: item["score"])[:show_layers]:
            lines.append(
                f"    layer={item['layer']} overlap={format_float(item['score'])} shared={item['overlap']}"
            )
    return lines


def main():
    parser = argparse.ArgumentParser(description="Summarize structured routed-MoE profile JSON artifacts.")
    parser.add_argument("profiles", nargs="+", help="Path(s) to --moe-profile-json artifacts")
    parser.add_argument("--show-layers", type=int, default=8, help="How many layer rows to print in each section")
    args = parser.parse_args()

    loaded = []
    for raw_path in args.profiles:
        path = Path(raw_path)
        data, lines = summarize_profile(path, args.show_layers)
        loaded.append((path, data))
        for line in lines:
            print(line)

    if len(loaded) >= 2:
        lhs_path, lhs_data = loaded[0]
        rhs_path, rhs_data = loaded[1]
        for line in compare_profiles(lhs_path, lhs_data, rhs_path, rhs_data, args.show_layers):
            print(line)


if __name__ == "__main__":
    main()
