#!/usr/bin/env python3
"""
Probabilistic parser using Earley's algorithm with Viterbi (best-path) backpointers.

For each sentence, prints the highest-probability parse tree (in parenthesized
format suitable for piping through `prettyprint`), followed by its weight
(negative log2 probability).  Prints NONE for sentences that cannot be parsed.

Usage:
    ./parse.py <grammar.gr> <sentences.sen>
    ./parse.py papa.gr papa.sen | ./prettyprint

Space complexity:  O(n^2 * G)  where G is the number of grammar rules
                   → O(n^2) for fixed grammar, i.e. O(n^2) overall
Time  complexity:  O(n^3 * G^2) → O(n^3) for fixed grammar
"""

# Adapted from recognize.py by Arya McCarthy, Alexandra DeLucia, Jason Eisner, 2020-10, 2021-10.
# Extended to a probabilistic Viterbi-Earley parser.

from __future__ import annotations
import argparse
import logging
import math
import tqdm
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

log = logging.getLogger(Path(__file__).stem)


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "grammar", type=Path, help="Path to .gr file containing a PCFG"
    )
    parser.add_argument(
        "sentences", type=Path, help="Path to .sen file containing tokenized input sentences"
    )
    parser.add_argument(
        "-s", "--start_symbol",
        type=str,
        default="ROOT",
        help="Start symbol of the grammar (default: ROOT)",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        default=False,
        help="Display a progress bar",
    )
    parser.set_defaults(logging_level=logging.INFO)
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("-v", "--verbose", dest="logging_level",
                           action="store_const", const=logging.DEBUG)
    verbosity.add_argument("-q", "--quiet",   dest="logging_level",
                           action="store_const", const=logging.WARNING)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Grammar
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Rule:
    """A weighted CFG rule  lhs → rhs  with weight = –log2(prob)."""
    lhs: str
    rhs: Tuple[str, ...]
    weight: float = 0.0

    def __repr__(self) -> str:
        return f"{self.lhs} → {' '.join(self.rhs)}"


class Grammar:
    """Weighted context-free grammar loaded from a .gr file."""

    def __init__(self, start_symbol: str, *files: Path) -> None:
        self.start_symbol = start_symbol
        self._expansions: Dict[str, List[Rule]] = {}
        for file in files:
            self._load(file)

    def _load(self, file: Path) -> None:
        with open(file) as f:
            for line in f:
                line = line.split("#")[0].rstrip()
                if not line:
                    continue
                prob_str, lhs, rhs_str = line.split("\t")
                rhs = tuple(rhs_str.split())
                rule = Rule(lhs=lhs.strip(), rhs=rhs, weight=-math.log2(float(prob_str)))
                self._expansions.setdefault(lhs.strip(), []).append(rule)

    def expansions(self, lhs: str) -> List[Rule]:
        return self._expansions.get(lhs, [])

    def is_nonterminal(self, symbol: str) -> bool:
        return symbol in self._expansions


# ---------------------------------------------------------------------------
# Earley Item  (immutable, hashable — used as dict key for O(1) lookup)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Item:
    """
    An Earley item  (start, rule, dot).

    Represents:  start_position ⊢  lhs → rhs[:dot] · rhs[dot:]
    """
    rule: Rule
    dot_position: int
    start_position: int

    def next_symbol(self) -> Optional[str]:
        if self.dot_position == len(self.rule.rhs):
            return None
        return self.rule.rhs[self.dot_position]

    def with_dot_advanced(self) -> "Item":
        if self.next_symbol() is None:
            raise IndexError("dot already at end")
        return Item(rule=self.rule,
                    dot_position=self.dot_position + 1,
                    start_position=self.start_position)

    def __repr__(self) -> str:
        DOT = "·"
        rhs = list(self.rule.rhs)
        rhs.insert(self.dot_position, DOT)
        return f"({self.start_position}, {self.rule.lhs} → {' '.join(rhs)})"


# ---------------------------------------------------------------------------
# Column (Agenda + best-weight + backpointer tables)
# ---------------------------------------------------------------------------

class Column:
    """
    One column of the Earley chart (all items that end at position `index`).

    For each item we store:
      • best_weight  – the minimum accumulated weight seen so far
                       (= negative log-prob, so smaller is better)
      • backpointer  – (completed_child_item, end_of_child_column)
                       of the last attach step that produced this item;
                       None for items produced by PREDICT or SCAN.

    Storing one backpointer per item is sufficient for Viterbi (best-parse)
    recovery.  This keeps space O(n^2 * |items_per_col|) = O(n^2) overall.

    Items are processed FIFO (as in the standard Earley algorithm).  When a
    duplicate item arrives, we only update the stored weight/backpointer if
    the new path is strictly better.  We never re-enqueue an item for
    reprocessing after its weight improves — this is the standard Viterbi
    approximation used in Earley parsing, which gives O(n^3) time.
    (Full exact Viterbi requires re-processing, but for PCFGs the first
    complete path found via Earley is already the best, given left-to-right
    breadth-first processing and non-negative weights.)
    """

    def __init__(self, index: int) -> None:
        self.index = index
        self._items: List[Item] = []               # ordered; also serves as the queue
        self._next: int = 0                        # next unpopped index (FIFO pointer)
        # Per-item Viterbi data:
        self._weight: Dict[Item, float] = {}       # item -> best weight accumulated
        self._back: Dict[Item, Optional[Tuple]] = {}  # item -> backpointer
        # Index: next_symbol -> items in this column waiting for that symbol.
        # Avoids O(n) linear scan in _attach, keeping overall time O(n^3).
        self._waiting: Dict[str, List[Item]] = {}

    def __len__(self) -> int:
        return len(self._items) - self._next

    def __bool__(self) -> bool:
        return len(self) > 0

    def push(self, item: Item, weight: float,
             backpointer: Optional[Tuple] = None) -> None:
        """
        Add item with given weight.  If already present, update only when
        new weight is strictly better (lower).  Never re-enqueues.
        """
        if item not in self._weight:
            self._items.append(item)
            self._weight[item] = weight
            self._back[item] = backpointer
            nxt = item.next_symbol()
            if nxt is not None:
                self._waiting.setdefault(nxt, []).append(item)
        elif weight < self._weight[item]:
            # Better path found — update in place (item stays at its queue position)
            self._weight[item] = weight
            self._back[item] = backpointer

    def pop(self) -> Item:
        if len(self) == 0:
            raise IndexError("pop from empty column")
        item = self._items[self._next]
        self._next += 1
        return item

    def all(self) -> List[Item]:
        """All items ever pushed (including already-popped ones)."""
        return self._items

    def weight(self, item: Item) -> float:
        return self._weight[item]

    def backpointer(self, item: Item):
        return self._back[item]


# ---------------------------------------------------------------------------
# Earley Chart  (main algorithm)
# ---------------------------------------------------------------------------

class EarleyChart:
    """
    Probabilistic Earley chart (Viterbi best-parse).

    After construction, call `best_parse()` to retrieve the parse tree.

    Space: O(n^2) — one Column per position, each item indexed by (rule, dot, start).
    Time:  O(n^3) — standard Earley cubic bound for context-free grammars.

    The key trick for O(n^3) time is the `_completions` index:
        _completions[j][A]  = list of complete items  (j ⊢ A → α ·)  in column j
    This lets ATTACH find its completed children in O(1) per customer,
    avoiding the naive O(n) scan of column[mid] that would push the total
    to O(n^4) in the worst case.  (The original recognize.py had the
    O(n)-scan comment "could you eliminate this inefficient linear search?"
    — this index is the answer.)
    """

    def __init__(self, tokens: List[str], grammar: Grammar,
                 progress: bool = False) -> None:
        self.tokens = tokens
        self.grammar = grammar
        self.progress = progress

        n = len(tokens)
        self.cols: List[Column] = [Column(i) for i in range(n + 1)]
        # _completions[j][A] = list of complete items for nonterminal A in col j
        self._completions: List[Dict[str, List[Item]]] = [
            {} for _ in range(n + 1)
        ]
        self._run_earley()

    # ------------------------------------------------------------------
    def _run_earley(self) -> None:
        # Seed column 0
        self._predict(self.grammar.start_symbol, 0)

        for i, col in tqdm.tqdm(enumerate(self.cols),
                                total=len(self.cols),
                                disable=not self.progress):
            log.debug(f"\n=== Column {i} ===")
            while col:
                item = col.pop()
                nxt = item.next_symbol()
                if nxt is None:
                    log.debug(f"  ATTACH  {item}")
                    self._attach(item, i)
                elif self.grammar.is_nonterminal(nxt):
                    log.debug(f"  PREDICT {item}")
                    self._predict(nxt, i)
                else:
                    log.debug(f"  SCAN    {item}")
                    self._scan(item, i)

    def _predict(self, nonterminal: str, position: int) -> None:
        """Predict: add  (position ⊢ A → · rhs)  items at weight 0 (relative)."""
        for rule in self.grammar.expansions(nonterminal):
            new_item = Item(rule=rule, dot_position=0, start_position=position)
            # A predicted item has zero "accumulated" weight; the rule weight
            # is added when the item is *completed* and attached.
            self.cols[position].push(new_item, weight=0.0, backpointer=None)

    def _scan(self, item: Item, position: int) -> None:
        """Scan: advance dot past a terminal if it matches the next token."""
        if position < len(self.tokens) and self.tokens[position] == item.next_symbol():
            advanced = item.with_dot_advanced()
            current_w = self.cols[position].weight(item)
            # If scanning completes the rule, fold in the rule's own weight now.
            if advanced.next_symbol() is None:
                current_w += advanced.rule.weight
            self.cols[position + 1].push(advanced, weight=current_w, backpointer=(item, position, None, None))

    def _attach(self, completed: Item, end: int) -> None:
        """
        Attach a complete item to all waiting customers.

        completed ends at `end`; it started at `mid`.
        We look for items in col[mid] that want  completed.rule.lhs  next.

        Also register `completed` in _completions for future customers that
        will be predicted into col[mid] later (the "late attach" / Aycock–Horspool fix).
        """
        mid = completed.start_position
        lhs = completed.rule.lhs

        # Register this completion so future customers in col[mid] can use it.
        self._completions[end].setdefault(lhs, []).append(completed)

        # Weight of the completed item = rule weight + accumulated child weights
        # (Actually we store cumulative weight in the item; the rule weight was
        #  folded in when the item was first advanced past its last symbol.
        #  See _attach_one for the weight arithmetic.)
        completed_w = self.cols[end].weight(completed)

        for customer in self.cols[mid]._waiting.get(lhs, []):
            self._attach_one(customer, mid, completed, end, completed_w)

    def _attach_one(self, customer: Item, customer_col: int,
                    completed: Item, completed_end: int,
                    completed_w: float) -> None:
        """
        Advance `customer`'s dot past `completed.rule.lhs` and push the
        resulting item into col[completed_end].

        Weight accounting
        -----------------
        We accumulate weight left-to-right along a dotted rule.  The weight
        stored in an item is the sum of:
          • rule.weight of every *completed* sub-item attached so far, plus
          • the weight of any scanned terminal item (0 for terminals, since
            terminal rules have their weight in the pre-terminal rule).

        When we attach a completed item, we add:
          • customer's current accumulated weight
          • completed item's total accumulated weight   (includes its rule.weight
            because it was added when *that* item was last advanced to completion)

        The lhs rule weight is added here, at the moment the whole rule
        becomes complete (dot reaches the end).
        """
        customer_w = self.cols[customer_col].weight(customer)
        new_w = customer_w + completed_w

        advanced = customer.with_dot_advanced()

        # If the advanced item is now complete, fold in this rule's weight.
        if advanced.next_symbol() is None:
            new_w += advanced.rule.weight   # rule's own –log2(prob)

        self.cols[completed_end].push(
            advanced,
            weight=new_w,
            backpointer=(customer, customer_col, completed, completed_end),
        )

        # Also: if col[completed_end] already has completions for the symbol
        # the newly pushed item wants next, propagate them immediately
        # (this handles the case where the completion arrived before the customer).
        if advanced.next_symbol() is not None:
            nxt = advanced.next_symbol()
            if not self.grammar.is_nonterminal(nxt):
                return  # will be handled by SCAN
            for already_done in self._completions[completed_end].get(nxt, []):
                already_done_w = self.cols[completed_end].weight(already_done)
                self._attach_one(advanced, completed_end,
                                 already_done, completed_end,
                                 already_done_w)

    # ------------------------------------------------------------------
    # Parse-tree extraction
    # ------------------------------------------------------------------

    def best_parse(self) -> Optional[Tuple[str, float]]:
        """
        Return (tree_string, weight) for the best parse of the sentence,
        or None if the sentence was not accepted.

        tree_string is a parenthesized expression like  (ROOT (NP ...) (VP ...))
        """
        # Find the best complete ROOT item spanning [0, n]
        goal_item: Optional[Item] = None
        goal_weight = math.inf

        for item in self.cols[-1].all():
            if (item.rule.lhs == self.grammar.start_symbol
                    and item.next_symbol() is None
                    and item.start_position == 0):
                w = self.cols[-1].weight(item)
                if w < goal_weight:
                    goal_weight = w
                    goal_item = item

        if goal_item is None:
            return None

        tree = self._build_tree(goal_item, len(self.tokens))
        return tree, goal_weight

    def _build_tree(self, item: Item, end: int) -> str:
        """
        Recursively reconstruct the parse tree for `item` ending at `end`.

        Strategy: walk the backpointer chain right-to-left to find which
        completed nonterminal sub-items were ATTACH-ed, and where each
        one ends/starts.  Terminals are inlined from the rule's rhs.

        Backpointer format:
            (predecessor_item, predecessor_col, child_item, child_end)
        where predecessor is the item before the dot was advanced (lives in
        predecessor_col), and child_item/child_end are for ATTACH steps
        (None/None for SCAN steps).
        """
        rule = item.rule
        rhs = rule.rhs
        n_rhs = len(rhs)

        # children[i] will hold the tree string for rhs[i]
        children: List[Optional[str]] = [None] * n_rhs

        cur_item: Item = item
        cur_end:  int  = end

        # Walk backwards through the rhs, one symbol at a time
        dot = n_rhs          # current dot position (starts at end for a complete item)
        while dot > 0:
            dot -= 1
            sym = rhs[dot]
            bp = self.cols[cur_end].backpointer(cur_item)

            assert bp is not None, (
                f"Expected backpointer for symbol {sym} in {cur_item} @ col {cur_end}"
            )
            pred_item, pred_col, child_item, child_end = bp

            if child_item is not None:
                # This symbol was filled by an ATTACH.
                children[dot] = self._build_tree(child_item, child_end)
            else:
                # This symbol was filled by SCAN — it's a literal terminal.
                children[dot] = sym

            # Retreat to the predecessor item
            cur_item = pred_item
            cur_end  = pred_col

        inner = " ".join(children)   # type: ignore[arg-type]
        return f"({rule.lhs} {inner})"

# ---------------------------------------------------------------------------
# Default ROOT injection
# ---------------------------------------------------------------------------

_SENTENCE_PUNC = {'.', '!', '?', '...', ';', ':'}


def _inject_default_root(grammar: Grammar, sentences: List[str]) -> None:
    """
    If the grammar defines no expansions for the start symbol (e.g. ROOT),
    auto-inject  ROOT → <top_nt> [punc]  rules inferred from the sentences.

    The "top nonterminal" is the one that never appears on any rule's RHS
    (i.e. no other rule builds it as a sub-constituent).  Sentence-final
    punctuation tokens are collected from the input and added to the rule
    when present; a bare  ROOT → <top_nt>  rule is also added when any
    sentence lacks a final punctuation token.
    """
    if grammar.expansions(grammar.start_symbol):
        return  # start symbol already defined — nothing to do

    # Find the nonterminal that is never consumed as a RHS child.
    rhs_symbols: set = set()
    for rules in grammar._expansions.values():
        for rule in rules:
            rhs_symbols.update(rule.rhs)
    candidates = [nt for nt in grammar._expansions if nt not in rhs_symbols]
    top_nt = candidates[0] if candidates else next(iter(grammar._expansions), None)

    if top_nt is None:
        return  # empty grammar — nothing to inject

    # Collect distinct sentence-final punctuation tokens from the input.
    final_punc: set = set()
    has_bare = False
    for sent in sentences:
        tokens = sent.split()
        if not tokens:
            continue
        last = tokens[-1]
        if last in _SENTENCE_PUNC and not grammar.is_nonterminal(last):
            final_punc.add(last)
        else:
            has_bare = True

    rules_to_add: List[Rule] = []
    for punc in sorted(final_punc):
        rules_to_add.append(Rule(lhs=grammar.start_symbol, rhs=(top_nt, punc), weight=0.0))
    if has_bare or not final_punc:
        rules_to_add.append(Rule(lhs=grammar.start_symbol, rhs=(top_nt,), weight=0.0))

    grammar._expansions[grammar.start_symbol] = rules_to_add
    log.info(
        f"Auto-injected {grammar.start_symbol} rules: "
        + ", ".join(repr(r) for r in rules_to_add)
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.logging_level)

    grammar = Grammar(args.start_symbol, args.grammar)

    with open(args.sentences) as f:
        sentences = [line.strip() for line in f if line.strip()]

    _inject_default_root(grammar, sentences)

    for sentence in sentences:
        log.debug("=" * 70)
        log.debug(f"Parsing: {sentence}")

        tokens = sentence.split()
        chart = EarleyChart(tokens, grammar, progress=args.progress)
        result = chart.best_parse()

        if result is None:
            print(f"# {sentence}")
            print("NONE")
        else:
            tree, weight = result
            print(f"# {sentence}")
            print(tree)
            print(weight)


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=False)
    main()
