"""
Functions related to Markov chains. This code was originally implemented in
https://github.com/agilescientific/striplog.

Author: Matt Hall, scienxlab.org
Licence: Apache 2.0

Copyright 2023 Matt Hall

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from collections import namedtuple

import numpy as np
import scipy.stats


def observations(seq_of_seqs, states, step=1, include_self=False):
    """
    Compute observation matrix.
    """
    O = np.zeros(tuple(states.size for _ in range(step+1)))
    for seq in seq_of_seqs:
        seq = np.array(seq)
        _, integer_seq = np.where(seq.reshape(-1, 1) == states)
        for idx in zip(*[integer_seq[n:] for n in range(step+1)]):
            if (not include_self) and (0 in np.diff(idx)):
                continue
            O[idx] += 1
    return O


def hollow_matrix(M):
    """
    Return hollow matrix (zeros on diagonal).

    Args
        M (ndarray): a 'square' ndarray.

    Returns
        ndarray. The same array with zeros on the diagonal.
    """
    s = M.shape[0]
    idx = np.unravel_index(np.arange(0, s**2, s + 1), M.shape)
    M[idx] = 0
    return M


def regularize(sequence, strings_are_states=False) -> tuple:
    """
    Turn a sequence or sequence of sequences into a tuple of
    the unique elements in the sequence(s), plus a sequence
    of sequences (sort of equivalent to `np.atleast_2d()`).

    Args
        sequence (list-like): A list-like container of either
            states, or of list-likes of states.
        strings_are_states (bool): True if the strings are
            themselves states (i.e. words or tokens) and not
            sequences of one-character states. For example,
            set to True if you provide something like:

                ['sst', 'mud', 'mud', 'sst', 'lst', 'lst']

    Returns
        tuple. A tuple of the unique states, and a sequence
            of sequences.
    """
    if strings_are_states:
        if isinstance(sequence[0], str):
            seq_of_seqs = [sequence]
        else:
            seq_of_seqs = sequence
    else:
        # Just try to iterate over the contents of the sequence.
        try:
            seq_of_seqs = [list(i) if len(i) > 1 else i for i in sequence]
        except TypeError:
            seq_of_seqs = [list(sequence)]

        # Annoyingly, still have to fix case of single sequence of
        # strings... this seems really hacky.
        if len(seq_of_seqs[0]) == 1:
            seq_of_seqs = [seq_of_seqs]

    # Now we know we have a sequence of sequences.
    uniques = set()
    for seq in seq_of_seqs:
        for i in seq:
            uniques.add(i)

    return np.array(sorted(uniques)), seq_of_seqs


class Markov_chain:

    def __init__(self,
                 observed_counts,
                 states=None,
                 step=1,
                 include_self=None,
                 ):
        """
        Initialize the Markov chain instance.

        Args:
            observed_counts (ndarray): A 2-D array representing the counts
                of change of state in the Markov Chain.
            states (array-like): An array-like representing the possible states
                of the Markov Chain. Must be in the same order as `observed
                counts`.
            step (int): The maximum step size, default 1.
            include_self (bool): Whether to include self-to-self transitions.
        """
        self.step = step
        self.observed_counts = np.atleast_2d(observed_counts).astype(int)

        if include_self is not None:
            self.include_self = include_self
        else:
            self.include_self = np.any(np.diagonal(self.observed_counts))

        if not self.include_self:
            self.observed_counts = hollow_matrix(self.observed_counts)

        if states is not None:
            self.states = np.asarray(states)
        else:
            self.states = np.arange(self.observed_counts.shape[0])

        if self.step > 1:
            self.expected_counts = self._compute_expected_mc()
        else:
            self.expected_counts = self._compute_expected()

        return

    @staticmethod
    def _compute_freqs(C):
        """
        Compute frequencies from counts.
        """
        epsilon = 1e-12
        return (C.T / (epsilon+np.sum(C.T, axis=0))).T

    @staticmethod
    def _stop_iter(a, b, tol=0.01):
        a_small = np.all(np.abs(a[-1] - a[-2]) < tol*a[-1])
        b_small = np.all(np.abs(b[-1] - b[-2]) < tol*b[-1])
        return (a_small and b_small)

    @property
    def _index_dict(self):
        if self.states is None:
            return {}
        return {self.states[index]: index for index in range(len(self.states))}

    @property
    def _state_dict(self):
        if self.states is None:
            return {}
        return {index: self.states[index] for index in range(len(self.states))}

    @property
    def observed_freqs(self):
        return self._compute_freqs(self.observed_counts)

    @property
    def expected_freqs(self):
        return self._compute_freqs(self.expected_counts)

    @property
    def _state_counts(self):
        s = self.observed_counts.copy()

        # Deal with more than 2 dimensions.
        for _ in range(self.observed_counts.ndim - 2):
            s = np.sum(s, axis=0)

        a = np.sum(s, axis=0)
        b = np.sum(s, axis=1)
        return np.maximum(a, b)

    @property
    def _state_probs(self):
        return self._state_counts / np.sum(self._state_counts)

    @property
    def normalized_difference(self):
        O = self.observed_counts
        E = self.expected_counts
        epsilon = 1e-12
        return (O - E) / np.sqrt(E + epsilon)

    @classmethod
    def from_sequence(cls,
                      sequence,
                      states=None,
                      strings_are_states=False,
                      include_self=False,
                      step=1,
                      ):
        """
        Parse a sequence and make the transition matrix of the specified order.

        **Provide sequence(s) ordered in upwards direction.**

        Args:
            sequence (list-like): A list-like, or list-like of list-likes.
                The inner list-likes represent sequences of states.
                For example, can be a string or list of strings, or
                a list or list of lists.
            states (list-like): A list or array of the names of the states.
                If not provided, it will be inferred from the data.
            strings_are_states (bool): True if the strings are
                themselves states (i.e. words or tokens) and not
                sequences of one-character states. For example,
                set to True if you provide something like:

                    ['sst', 'mud', 'mud', 'sst', 'lst', 'lst']

            include_self (bool): Whether to include self-to-self
                transitions (default is `False`: do not include them).
            step (integer): The distance to step. Default is 1: use
                the previous state only. If 2, then the previous-but-
                one state is used as well as the previous state (and
                the matrix has one more dimension).
        """
        uniques, seq_of_seqs = regularize(sequence, strings_are_states=strings_are_states)

        if states is None:
            states = uniques
        else:
            states = np.asarray(list(states))

        O = observations(seq_of_seqs, states=states, step=step, include_self=include_self)

        return cls(observed_counts=np.array(O),
                   states=states,
                   include_self=include_self,
                   step=step,
                   )

    def _conditional_probs(self, state):
        """
        Conditional probabilities of each state, given a
        current state.
        """
        return self.observed_freqs[self._index_dict[state]]

    def _next_state(self, current_state):
        """
        Returns the state of the random variable at the next time
        instance.

        Args:
            current_state (str): The current state of the system.

        Returns:
            str. One realization of the next state.
        """
        return np.random.choice(self.states,
                                p=self._conditional_probs(current_state)
                                )

    def generate_states(self, n=10, current_state=None):
        """
        Generates the next states of the system.

        Args:
            n (int): The number of future states to generate.
            current_state (str): The state of the current random variable.

        Returns:
            list. The next n states.
        """
        if current_state is None:
            current_state = np.random.choice(self.states, p=self._state_probs)

        future_states = []
        for _ in range(n):
            next_state = self._next_state(current_state)
            future_states.append(next_state)
            current_state = next_state

        return future_states

    def _compute_expected(self):
        """
        Try to use Powers & Easterling, fall back on Monte Carlo sampling
        based on the proportions of states in the data.
        """
        try:
            E = self._compute_expected_pe()
        except:
            E = self._compute_expected_mc()

        return E

    def _compute_expected_mc(self, n=100000):
        """
        If we can't use Powers & Easterling's method, and it's possible there's
        a way to extend it to higher dimensions (which we have for step > 1),
        the next best thing might be to use brute force and just compute a lot
        of random sequence transitions, given the observed proportions. This is
        what P & E's method tries to estimate iteratively.

        What to do about 'self transitions' is a bit of a problem here, since
        there are a lot of n-grams that include at least one self-transition.
        """
        seq = np.random.choice(self.states, size=n, p=self._state_probs)
        E = observations(np.atleast_2d(seq), self.states, step=self.step, include_self=self.include_self)
        if not self.include_self:
            E = hollow_matrix(E)
        return np.sum(self.observed_counts) * E / np.sum(E)

    def _compute_expected_pe(self, max_iter=100):
        """
        Compute the independent trials matrix, using method of
        Powers & Easterling 1982.
        """
        m = len(self.states)
        M = self.observed_counts
        a, b = [], []

        # Loop 1
        a.append(np.sum(M, axis=1) / (m - 1))
        b.append(np.sum(M, axis=0) / (np.sum(a[-1]) - a[-1]))

        i = 2
        while i < max_iter:

            a.append(np.sum(M, axis=1) / (np.sum(b[-1]) - b[-1]))
            b.append(np.sum(M, axis=0) / (np.sum(a[-1]) - a[-1]))

            # Check for stopping criterion.
            if self._stop_iter(a, b, tol=0.001):
                break

            i += 1

        E = a[-1] * b[-1].reshape(-1, 1)

        if not self.include_self:
            return hollow_matrix(E)
        else:
            return E

    @property
    def degrees_of_freedom(self) -> int:
        m = len(self.states)
        return (m - 1)**2 - m

    def _chi_squared_critical(self, q=0.95, df=None):
        """
        The chi-squared critical value for a confidence level q
        and degrees of freedom df.
        """
        if df is None:
            df = self.degrees_of_freedom
        return scipy.stats.chi2.ppf(q=q, df=df)

    def _chi_squared_percentile(self, x, df=None):
        """
        The chi-squared critical value for a confidence level q
        and degrees of freedom df.
        """
        if df is None:
            df = self.degrees_of_freedom
        return scipy.stats.chi2.cdf(x, df=df)

    def chi_squared(self, q=0.95):
        """
        The chi-squared statistic for the given transition
        frequencies.

        Also returns the critical statistic at the given confidence
        level q (default 95%).

        If the first number is bigger than the second number,
        then you can reject the hypothesis that the sequence
        is randomly ordered.

        Args:
            q (float): The confidence level, as a float in the range 0 to 1.
                Default: 0.95.

        Returns:
            float: The chi-squared statistic.
        """
        # Observed and Expected matrices:
        O = self.observed_counts
        E = self.expected_counts

        # Adjustment for divide-by-zero
        epsilon = 1e-12
        chi2 = np.sum((O - E)**2 / (E + epsilon))
        crit = self._chi_squared_critical(q=q)
        perc = self._chi_squared_percentile(x=chi2)
        Chi2 = namedtuple('Chi2', ['chi2', 'crit', 'perc'])

        return Chi2(chi2, crit, perc)
