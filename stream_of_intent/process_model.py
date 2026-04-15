from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from pm4py.algo.simulation.playout.process_tree import algorithm as pt_playout
from pm4py.algo.simulation.tree_generator import algorithm as tree_gen
from pm4py.algo.simulation.tree_generator.variants import ptandloggenerator
from pm4py.objects.process_tree.obj import Operator, ProcessTree

from stream_of_intent.types import ModelParams

_START_SYMBOL = "__START__"
_END_SYMBOL = "__END__"


class ProcessModel:
    """Wrapper around a pm4py ProcessTree with enriched transition data.

    Holds the process tree, a sample of traces generated from it, and
    variable-order transition matrices extracted from those traces.

    Attributes:
        tree: The underlying pm4py ProcessTree.
        activities: Sorted list of activity labels in the model.
        traces: List of sampled traces (each a list of activity strings).
    """

    def __init__(
        self,
        tree: ProcessTree,
        num_sample_traces: int = 200,
        random_seed: Optional[int] = None,
    ) -> None:
        """Initialize a ProcessModel from a pm4py ProcessTree.

        Generates sample traces from the tree and extracts activity labels.

        Args:
            tree: A pm4py ProcessTree defining the control-flow structure.
            num_sample_traces: Number of traces to sample for building
                transition statistics.
            random_seed: Seed for reproducible trace generation.
        """
        self.tree = tree
        self._random_seed = random_seed
        self.traces = self._sample_traces(num_sample_traces)
        self.activities = sorted(
            {act for trace in self.traces for act in trace}
        )
        self._transition_matrices: Dict[int, Dict[Tuple[str, ...], Dict[str, float]]] = {}

    def _sample_traces(self, n: int) -> List[List[str]]:
        """Sample traces from the process tree via pm4py playout.

        Args:
            n: Number of traces to generate.

        Returns:
            List of traces, each being a list of activity label strings.
        """
        log = pt_playout.apply(
            self.tree,
            parameters={
                pt_playout.Variants.BASIC_PLAYOUT.value.Parameters.NO_TRACES: n,
            },
        )
        traces: List[List[str]] = []
        for trace in log:
            acts = [event["concept:name"] for event in trace]
            traces.append(acts)
        return traces

    def get_transition_matrix(
        self, order: int,
    ) -> Dict[Tuple[str, ...], Dict[str, float]]:
        """Get or build a variable-order transition probability matrix.

        Builds the matrix from sampled traces on first call and caches it.
        The matrix maps a context tuple of ``order`` preceding activities
        to a distribution over next activities.

        The special context starting with ``__START__`` symbols represents
        the beginning of a trace, and ``__END__`` represents trace completion.

        Args:
            order: Number of preceding activities to condition on.

        Returns:
            Dictionary mapping context tuples to dictionaries of
            {next_activity: probability}.
        """
        if order in self._transition_matrices:
            return self._transition_matrices[order]

        context_counts: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)

        for trace in self.traces:
            padded = [_START_SYMBOL] * order + trace + [_END_SYMBOL]
            for i in range(order, len(padded)):
                context = tuple(padded[i - order:i])
                context_counts[context][padded[i]] += 1

        matrix: Dict[Tuple[str, ...], Dict[str, float]] = {}
        for context, counts in context_counts.items():
            total = sum(counts.values())
            matrix[context] = {
                act: count / total for act, count in counts.items()
            }

        self._transition_matrices[order] = matrix
        return matrix

    def get_subprocess_model(
        self,
        rng: np.random.RandomState,
        simplify: bool = True,
    ) -> ProcessModel:
        """Derive a child subprocess model from this model.

        Creates a structurally simplified version of the process tree
        suitable for subprocess spawning (fractal behavior).  The child
        model uses a subset of activities and reduced nesting.

        Args:
            rng: Random state for reproducible subprocess derivation.
            simplify: Whether to structurally simplify the tree. If False,
                returns a copy of the same model.

        Returns:
            A new ProcessModel for the subprocess.
        """
        if not simplify or len(self.activities) <= 2:
            return ProcessModel(self.tree, num_sample_traces=50)

        subset_size = max(2, len(self.activities) // 2)
        subset = list(rng.choice(self.activities, size=subset_size, replace=False))

        child_tree = _build_sequence_tree(subset)
        return ProcessModel(child_tree, num_sample_traces=50)


class ProcessModelGenerator:
    """Generates process models from tunable parameters.

    Uses pm4py's PTAndLogGenerator to create random process trees with
    operator probabilities controlled by :class:`ModelParams`.
    """

    def generate(self, params: ModelParams, seed: Optional[int] = None) -> ProcessModel:
        """Generate a process model from the given parameters.

        Args:
            params: Tunable parameters controlling the process tree structure
                (operator weights, number of activities, nesting depth, etc.).
            seed: Random seed for reproducible generation.

        Returns:
            A ProcessModel wrapping the generated process tree.
        """
        total_weight = (
            params.sequence_weight
            + params.parallel_weight
            + params.loop_weight
            + params.choice_weight
        )
        if total_weight < 1e-12:
            total_weight = 1.0

        gen_params = {
            ptandloggenerator.Parameters.MIN: max(3, params.num_activities - 2),
            ptandloggenerator.Parameters.MAX: params.num_activities + 2,
            ptandloggenerator.Parameters.MODE: params.num_activities,
            ptandloggenerator.Parameters.SEQUENCE: params.sequence_weight / total_weight,
            ptandloggenerator.Parameters.CHOICE: params.choice_weight / total_weight,
            ptandloggenerator.Parameters.PARALLEL: params.parallel_weight / total_weight,
            ptandloggenerator.Parameters.LOOP: params.loop_weight / total_weight,
            ptandloggenerator.Parameters.SILENT: params.silent_transition_probability,
        }

        tree = tree_gen.apply(
            parameters=gen_params,
            variant=tree_gen.Variants.PTANDLOGGENERATOR,
        )

        if params.nesting_depth > 0:
            _limit_depth(tree, params.nesting_depth)

        return ProcessModel(tree, num_sample_traces=200, random_seed=seed)


def _limit_depth(node: ProcessTree, max_depth: int, current_depth: int = 0) -> None:
    """Recursively limit the depth of a process tree.

    Nodes beyond ``max_depth`` are collapsed to leaf nodes using the first
    activity label found in their subtree.

    Args:
        node: Current node in the tree.
        max_depth: Maximum allowed depth.
        current_depth: Current depth in the recursion.
    """
    if current_depth >= max_depth and node.children:
        first_label = _first_leaf_label(node)
        node.children = []
        node._operator = None
        node._label = first_label

    for child in node.children:
        _limit_depth(child, max_depth, current_depth + 1)


def _first_leaf_label(node: ProcessTree) -> Optional[str]:
    """Find the first leaf label in a subtree.

    Args:
        node: Root of the subtree to search.

    Returns:
        The label of the first leaf found, or None if all are silent.
    """
    if not node.children:
        return node.label
    for child in node.children:
        label = _first_leaf_label(child)
        if label is not None:
            return label
    return None


def _build_sequence_tree(activities: List[str]) -> ProcessTree:
    """Build a simple sequential process tree from a list of activities.

    Args:
        activities: List of activity labels to place in sequence.

    Returns:
        A pm4py ProcessTree with a SEQUENCE operator over the activities.
    """
    if len(activities) == 1:
        return ProcessTree(label=activities[0])

    root = ProcessTree(operator=Operator.SEQUENCE)
    for act in activities:
        child = ProcessTree(label=act, parent=root)
        root.children.append(child)
    return root
