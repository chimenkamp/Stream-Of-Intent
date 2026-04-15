from __future__ import annotations

import math
from collections import Counter, defaultdict
from itertools import islice
from typing import Dict, Iterator, List, Sequence, Tuple

import numpy as np

from stream_of_intent.types import Event, FeatureVector


def _group_by_case(window: Sequence[Event]) -> Dict[str, List[str]]:
    """Group end-event activity labels by case_id, preserving order.

    Args:
        window: Sequence of events in a single tumbling window.

    Returns:
        Dictionary mapping case_id to ordered list of activity labels.
    """
    case_activities: Dict[str, List[str]] = defaultdict(list)
    for e in window:
        if e.event_type == "end":
            case_activities[e.case_id].append(e.activity)
    return dict(case_activities)


def temporal_dependency_strength(window: Sequence[Event]) -> float:
    """Measure short-range sequential predictability via entropy ratio.

    Computes how predictable the next activity is given only the immediately
    preceding activity *within each case*, then aggregates across cases
    weighted by the number of transitions.  This avoids the dilution caused
    by interleaving events from multiple concurrent cases.

    The returned strength is ``1 - ratio``, so 1.0 = perfectly predictable,
    0.0 = no temporal dependency.

    Args:
        window: Sequence of events in a single tumbling window.

    Returns:
        Temporal dependency strength in [0, 1].
    """
    case_activities = _group_by_case(window)

    total_weight = 0
    weighted_strength = 0.0

    for activities in case_activities.values():
        if len(activities) < 2:
            continue

        marginal_entropy = _entropy_of_sequence(activities)
        if marginal_entropy < 1e-12:
            case_strength = 1.0
        else:
            bigram_counts: Dict[str, Counter] = defaultdict(Counter)
            for prev_act, curr_act in zip(activities[:-1], activities[1:]):
                bigram_counts[prev_act][curr_act] += 1

            conditional_entropy = 0.0
            total_transitions = len(activities) - 1
            for prev_act, next_counts in bigram_counts.items():
                prev_total = sum(next_counts.values())
                weight = prev_total / total_transitions
                h = _entropy_of_counter(next_counts)
                conditional_entropy += weight * h

            ratio = conditional_entropy / marginal_entropy
            case_strength = float(np.clip(1.0 - ratio, 0.0, 1.0))

        num_transitions = len(activities) - 1
        weighted_strength += case_strength * num_transitions
        total_weight += num_transitions

    if total_weight == 0:
        return 0.0

    return float(np.clip(weighted_strength / total_weight, 0.0, 1.0))


def long_term_dependency_strength(window: Sequence[Event]) -> float:
    """Measure long-range dependency via higher-order Markov model comparison.

    Compares the predictive power of an order-1 Markov model with an order-k
    model (k=3 by default) *within each case*, then aggregates across cases.
    If the higher-order model significantly reduces conditional entropy, the
    stream has strong long-term dependencies.

    The strength is the normalized reduction in conditional entropy:
    ``(H_1 - H_k) / H_1`` where H_n is the conditional entropy of the
    order-n model.

    Args:
        window: Sequence of events in a single tumbling window.

    Returns:
        Long-term dependency strength in [0, 1].
    """
    case_activities = _group_by_case(window)
    max_order = 3

    total_weight = 0
    weighted_strength = 0.0

    for activities in case_activities.values():
        if len(activities) < max_order + 2:
            continue

        h1 = _conditional_entropy_order_n(activities, 1)
        hk = _conditional_entropy_order_n(activities, max_order)

        if h1 < 1e-12:
            continue

        reduction = (h1 - hk) / h1
        case_strength = float(np.clip(reduction, 0.0, 1.0))

        num_transitions = len(activities) - max_order
        weighted_strength += case_strength * num_transitions
        total_weight += num_transitions

    if total_weight == 0:
        return 0.0

    return float(np.clip(weighted_strength / total_weight, 0.0, 1.0))


def non_linear_dependency_strength(window: Sequence[Event]) -> float:
    """Measure branching, parallelism, and context-sensitivity.

    Combines three sub-measures:
    1. **Branching factor**: average number of distinct successors per activity,
       computed per-case and aggregated to avoid cross-case inflation.
    2. **Parallelism degree**: fraction of time intervals where multiple cases
       have overlapping active events (stream-level).
    3. **Context sensitivity**: how much the successor distribution changes
       depending on the 2nd-to-last activity, computed per-case.

    These three components are averaged to yield the final score.

    Args:
        window: Sequence of events in a single tumbling window.

    Returns:
        Non-linear dependency strength in [0, 1].
    """
    case_activities = _group_by_case(window)

    branching = _weighted_branching_factor(case_activities)
    parallelism = _parallelism_degree(window)
    context = _weighted_context_sensitivity(case_activities)

    return float(np.clip((branching + parallelism + context) / 3.0, 0.0, 1.0))


def _weighted_branching_factor(
    case_activities: Dict[str, List[str]],
) -> float:
    """Compute per-case branching factor weighted by number of transitions."""
    total_weight = 0
    weighted_sum = 0.0
    for activities in case_activities.values():
        case_unique = len(set(activities))
        if len(activities) >= 2 and case_unique > 1:
            b = _branching_factor(activities, case_unique)
            n = len(activities) - 1
            weighted_sum += b * n
            total_weight += n
    return weighted_sum / total_weight if total_weight > 0 else 0.0


def _weighted_context_sensitivity(
    case_activities: Dict[str, List[str]],
) -> float:
    """Compute per-case context sensitivity weighted by number of trigrams."""
    total_weight = 0
    weighted_sum = 0.0
    for activities in case_activities.values():
        if len(activities) >= 3:
            c = _context_sensitivity(activities)
            n = len(activities) - 2
            weighted_sum += c * n
            total_weight += n
    return weighted_sum / total_weight if total_weight > 0 else 0.0


def out_of_order_strength(window: Sequence[Event]) -> float:
    """Measure the fraction and magnitude of out-of-order events.

    An event is out-of-order if its ``timestamp`` order disagrees with its
    ``arrival_timestamp`` order relative to any preceding event in the stream.

    The strength combines the fraction of out-of-order events with the
    normalized mean displacement magnitude.

    Args:
        window: Sequence of events in a single tumbling window.

    Returns:
        Out-of-order strength in [0, 1].
    """
    if len(window) < 2:
        return 0.0

    ooo_count = 0
    total_displacement = 0.0
    max_possible_displacement = 0.0

    timestamps = [e.timestamp for e in window]
    arrival_times = [e.arrival_timestamp for e in window]

    t_range = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 1.0
    if t_range < 1e-12:
        t_range = 1.0

    running_max_timestamp = timestamps[0]

    for i in range(1, len(window)):
        if timestamps[i] < running_max_timestamp:
            ooo_count += 1
            displacement = running_max_timestamp - timestamps[i]
            total_displacement += displacement
        running_max_timestamp = max(running_max_timestamp, timestamps[i])

    n = len(window)
    fraction = ooo_count / (n - 1) if n > 1 else 0.0

    mean_displacement = total_displacement / max(ooo_count, 1)
    normalized_displacement = min(mean_displacement / t_range, 1.0)

    strength = 0.6 * fraction + 0.4 * normalized_displacement
    return float(np.clip(strength, 0.0, 1.0))


def fractal_behavior_strength(window: Sequence[Event]) -> float:
    """Measure self-similar nested subprocess patterns via DFA and nesting.

    Combines two indicators:
    1. **Nesting depth**: detects hierarchical case relationships by examining
       case ID structure (parent/child naming convention) and temporal
       containment of cases.
    2. **Detrended Fluctuation Analysis (DFA)**: estimates the Hurst exponent
       of the inter-event time series. A Hurst exponent > 0.5 indicates
       long-range correlations (self-similarity), with H ≈ 1.0 indicating
       strong fractal behavior.

    Args:
        window: Sequence of events in a single tumbling window.

    Returns:
        Fractal behavior strength in [0, 1].
    """
    nesting = _nesting_depth_score(window)
    hurst = _dfa_hurst_exponent(window)

    return float(np.clip(0.5 * nesting + 0.5 * hurst, 0.0, 1.0))


def extract_features(window: Sequence[Event]) -> FeatureVector:
    """Compute all five feature values for a single event window.

    Args:
        window: Sequence of events in one tumbling window.

    Returns:
        A FeatureVector with all five features normalized to [0, 1].
    """
    return FeatureVector(
        temporal_dependency=temporal_dependency_strength(window),
        long_term_dependency=long_term_dependency_strength(window),
        non_linear_dependency=non_linear_dependency_strength(window),
        out_of_order=out_of_order_strength(window),
        fractal_behavior=fractal_behavior_strength(window),
    )


def tumbling_windows(
    stream: Iterator[Event],
    window_size: int,
) -> Iterator[List[Event]]:
    """Split an event stream into non-overlapping fixed-size windows.

    Consumes events from the stream iterator and yields lists of exactly
    ``window_size`` events.  The final partial window (if any) is discarded.

    Args:
        stream: An iterator yielding events ordered by arrival_timestamp.
        window_size: Number of events per window.

    Yields:
        Lists of ``window_size`` events forming each tumbling window.
    """
    while True:
        window = list(islice(stream, window_size))
        if len(window) < window_size:
            break
        yield window


def extract_features_windowed(
    stream: Iterator[Event],
    window_size: int,
    max_windows: int = 0,
) -> Iterator[FeatureVector]:
    """Extract features from each tumbling window of an event stream.

    Args:
        stream: An iterator yielding events ordered by arrival_timestamp.
        window_size: Number of events per window.
        max_windows: Maximum number of windows to process. 0 for unlimited.

    Yields:
        A FeatureVector for each tumbling window.
    """
    count = 0
    for window in tumbling_windows(stream, window_size):
        yield extract_features(window)
        count += 1
        if max_windows > 0 and count >= max_windows:
            break


def _entropy_of_sequence(seq: Sequence[str]) -> float:
    """Shannon entropy of a discrete sequence.

    Args:
        seq: Sequence of string labels.

    Returns:
        Entropy in nats.
    """
    counts = Counter(seq)
    return _entropy_of_counter(counts)


def _entropy_of_counter(counts: Counter) -> float:
    """Shannon entropy from a Counter of label frequencies.

    Args:
        counts: Counter mapping labels to occurrence counts.

    Returns:
        Entropy in nats.
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log(p)
    return entropy


def _conditional_entropy_order_n(
    activities: List[str],
    order: int,
) -> float:
    """Conditional entropy H(A_t | A_{t-1}, ..., A_{t-n}) for order-n Markov.

    Args:
        activities: Sequence of activity labels.
        order: Number of preceding activities to condition on.

    Returns:
        Conditional entropy in nats.
    """
    if len(activities) <= order:
        return 0.0

    context_counts: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)
    for i in range(order, len(activities)):
        context = tuple(activities[i - order:i])
        context_counts[context][activities[i]] += 1

    total_transitions = len(activities) - order
    conditional_entropy = 0.0
    for context, next_counts in context_counts.items():
        ctx_total = sum(next_counts.values())
        weight = ctx_total / total_transitions
        h = _entropy_of_counter(next_counts)
        conditional_entropy += weight * h

    return conditional_entropy


def _branching_factor(
    activities: List[str],
    num_unique: int,
) -> float:
    """Normalized average branching factor.

    Args:
        activities: Sequence of activity labels.
        num_unique: Number of distinct activities.

    Returns:
        Branching factor normalized to [0, 1].
    """
    if num_unique <= 1 or len(activities) < 2:
        return 0.0

    successors: Dict[str, set] = defaultdict(set)
    for prev_act, curr_act in zip(activities[:-1], activities[1:]):
        successors[prev_act].add(curr_act)

    if not successors:
        return 0.0

    avg_successors = np.mean([len(s) for s in successors.values()])
    return float(np.clip((avg_successors - 1.0) / max(num_unique - 1, 1), 0.0, 1.0))


def _parallelism_degree(window: Sequence[Event]) -> float:
    """Fraction of event time intervals with multiple concurrent activities.

    Args:
        window: Sequence of events in a single window.

    Returns:
        Parallelism degree normalized to [0, 1].
    """
    start_events: Dict[Tuple[str, str], float] = {}
    intervals: List[Tuple[float, float]] = []

    for e in window:
        key = (e.case_id, e.activity)
        if e.event_type == "start":
            start_events[key] = e.timestamp
        elif e.event_type == "end" and key in start_events:
            intervals.append((start_events[key], e.timestamp))
            del start_events[key]

    if len(intervals) < 2:
        return 0.0

    overlap_count = 0
    total_pairs = 0
    for i in range(len(intervals)):
        for j in range(i + 1, len(intervals)):
            total_pairs += 1
            s1, e1 = intervals[i]
            s2, e2 = intervals[j]
            if s1 < e2 and s2 < e1:
                overlap_count += 1

    return float(overlap_count / total_pairs) if total_pairs > 0 else 0.0


def _context_sensitivity(activities: List[str]) -> float:
    """Measure how the successor distribution changes based on 2-step history.

    Computes the Jensen-Shannon divergence between successor distributions
    conditioned on different 2nd-to-last activities, normalized to [0, 1].

    Args:
        activities: Sequence of activity labels.

    Returns:
        Context sensitivity score in [0, 1].
    """
    if len(activities) < 3:
        return 0.0

    trigram_counts: Dict[str, Dict[str, Counter]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    for i in range(2, len(activities)):
        prev2 = activities[i - 2]
        prev1 = activities[i - 1]
        trigram_counts[prev1][prev2][activities[i]] += 1

    divergences: List[float] = []
    for prev1, by_prev2 in trigram_counts.items():
        contexts = list(by_prev2.values())
        if len(contexts) < 2:
            continue
        for i in range(len(contexts)):
            for j in range(i + 1, len(contexts)):
                div = _jensen_shannon_divergence(contexts[i], contexts[j])
                divergences.append(div)

    if not divergences:
        return 0.0

    return float(np.clip(np.mean(divergences), 0.0, 1.0))


def _jensen_shannon_divergence(c1: Counter, c2: Counter) -> float:
    """Jensen-Shannon divergence between two distributions given as Counters.

    Args:
        c1: First distribution as a Counter.
        c2: Second distribution as a Counter.

    Returns:
        JSD in [0, 1] (normalized by log(2)).
    """
    all_keys = set(c1.keys()) | set(c2.keys())
    if not all_keys:
        return 0.0

    t1 = sum(c1.values())
    t2 = sum(c2.values())
    if t1 == 0 or t2 == 0:
        return 0.0

    jsd = 0.0
    for key in all_keys:
        p = c1.get(key, 0) / t1
        q = c2.get(key, 0) / t2
        m = (p + q) / 2.0
        if p > 0 and m > 0:
            jsd += 0.5 * p * math.log(p / m)
        if q > 0 and m > 0:
            jsd += 0.5 * q * math.log(q / m)

    max_jsd = math.log(2)
    return float(np.clip(jsd / max_jsd, 0.0, 1.0))


def _nesting_depth_score(window: Sequence[Event]) -> float:
    """Score subprocess nesting depth from case ID hierarchy.

    Detects parent-child case relationships via the naming convention
    ``parent_id.child_index`` and computes the normalized maximum depth.

    Args:
        window: Sequence of events in a single window.

    Returns:
        Nesting depth score in [0, 1].
    """
    case_ids = set(e.case_id for e in window)
    max_depth = 0
    for cid in case_ids:
        depth = cid.count(".")
        max_depth = max(max_depth, depth)

    return float(np.clip(max_depth / 4.0, 0.0, 1.0))


def _dfa_hurst_exponent(window: Sequence[Event]) -> float:
    """Estimate Hurst exponent via Detrended Fluctuation Analysis.

    Applied to the inter-event timestamp series.  A Hurst exponent H > 0.5
    indicates long-range correlations; H ≈ 0.5 indicates no correlation
    (white noise); H < 0.5 indicates anti-correlation.

    The result is mapped to [0, 1] by ``2 * (H - 0.5)`` clamped.

    Args:
        window: Sequence of events in a single window.

    Returns:
        Fractal strength based on Hurst exponent, in [0, 1].
    """
    timestamps = sorted(e.timestamp for e in window)
    if len(timestamps) < 16:
        return 0.0

    intervals = np.diff(timestamps)
    if len(intervals) < 16:
        return 0.0

    if np.std(intervals) < 1e-12:
        return 0.0

    cumulative = np.cumsum(intervals - np.mean(intervals))
    n = len(cumulative)

    min_box = 4
    max_box = n // 4
    if max_box <= min_box:
        return 0.0

    box_sizes = np.unique(
        np.logspace(
            np.log10(min_box),
            np.log10(max_box),
            num=min(20, max_box - min_box + 1),
        ).astype(int)
    )
    box_sizes = box_sizes[box_sizes >= min_box]

    if len(box_sizes) < 2:
        return 0.0

    fluctuations: List[float] = []
    valid_boxes: List[int] = []

    for box_size in box_sizes:
        num_boxes = n // box_size
        if num_boxes < 1:
            continue

        rms_values: List[float] = []
        for b in range(num_boxes):
            segment = cumulative[b * box_size:(b + 1) * box_size]
            x = np.arange(box_size)
            coeffs = np.polyfit(x, segment, 1)
            trend = np.polyval(coeffs, x)
            rms = np.sqrt(np.mean((segment - trend) ** 2))
            rms_values.append(rms)

        mean_rms = np.mean(rms_values)
        if mean_rms > 1e-12:
            fluctuations.append(mean_rms)
            valid_boxes.append(box_size)

    if len(valid_boxes) < 2:
        return 0.0

    log_boxes = np.log(np.array(valid_boxes, dtype=float))
    log_fluct = np.log(np.array(fluctuations, dtype=float))

    coeffs = np.polyfit(log_boxes, log_fluct, 1)
    hurst = coeffs[0]

    strength = 2.0 * (hurst - 0.5)
    return float(np.clip(strength, 0.0, 1.0))
