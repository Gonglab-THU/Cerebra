# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Common utilities for data pipeline tools."""
import contextlib
import datetime
import shutil
import tempfile
import time

import glob
import logging
import os
import subprocess

from concurrent import futures
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, Dict, Iterable, List
from urllib import request

import collections
import dataclasses
import re
import string



DeletionMatrix = Sequence[Sequence[int]]


@dataclasses.dataclass(frozen=True)
class TemplateHit:
    """Class representing a template hit."""

    index: int
    name: str
    aligned_cols: int
    sum_probs: float
    query: str
    hit_sequence: str
    indices_query: List[int]
    indices_hit: List[int]


def parse_fasta(fasta_string: str) -> Tuple[Sequence[str], Sequence[str]]:
    """Parses FASTA string and returns list of strings with amino-acid sequences.

    Arguments:
        fasta_string: The string contents of a FASTA file.

    Returns:
        A tuple of two lists:
        * A list of sequences.
        * A list of sequence descriptions taken from the comment lines. In the
            same order as the sequences.
    """
    sequences = []
    descriptions = []
    index = -1
    for line in fasta_string.splitlines():
        line = line.strip()
        if line.startswith(">"):
            index += 1
            descriptions.append(line[1:])  # Remove the '>' at the beginning.
            sequences.append("")
            continue
        elif line.startswith("#"):
            continue
        elif not line:
            continue  # Skip blank lines.
        sequences[index] += line

    return sequences, descriptions


def parse_stockholm(
    stockholm_string: str,
) -> Tuple[Sequence[str], DeletionMatrix, Sequence[str]]:
    """Parses sequences and deletion matrix from stockholm format alignment.

    Args:
        stockholm_string: The string contents of a stockholm file. The first
            sequence in the file should be the query sequence.

    Returns:
        A tuple of:
            * A list of sequences that have been aligned to the query. These
                might contain duplicates.
            * The deletion matrix for the alignment as a list of lists. The element
                at `deletion_matrix[i][j]` is the number of residues deleted from
                the aligned sequence i at residue position j.
            * The names of the targets matched, including the jackhmmer subsequence
                suffix.
    """
    name_to_sequence = collections.OrderedDict()
    for line in stockholm_string.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "//")):
            continue
        name, sequence = line.split()
        if name not in name_to_sequence:
            name_to_sequence[name] = ""
        name_to_sequence[name] += sequence

    msa = []
    deletion_matrix = []

    query = ""
    keep_columns = []
    for seq_index, sequence in enumerate(name_to_sequence.values()):
        if seq_index == 0:
            # Gather the columns with gaps from the query
            query = sequence
            keep_columns = [i for i, res in enumerate(query) if res != "-"]

        # Remove the columns with gaps in the query from all sequences.
        aligned_sequence = "".join([sequence[c] for c in keep_columns])

        msa.append(aligned_sequence)

        # Count the number of deletions w.r.t. query.
        deletion_vec = []
        deletion_count = 0
        for seq_res, query_res in zip(sequence, query):
            if seq_res != "-" or query_res != "-":
                if query_res == "-":
                    deletion_count += 1
                else:
                    deletion_vec.append(deletion_count)
                    deletion_count = 0
        deletion_matrix.append(deletion_vec)

    return msa, deletion_matrix, list(name_to_sequence.keys())


def parse_a3m(a3m_string: str) -> Tuple[Sequence[str], DeletionMatrix]:
    """Parses sequences and deletion matrix from a3m format alignment.

    Args:
        a3m_string: The string contents of a a3m file. The first sequence in the
            file should be the query sequence.

    Returns:
        A tuple of:
            * A list of sequences that have been aligned to the query. These
                might contain duplicates.
            * The deletion matrix for the alignment as a list of lists. The element
                at `deletion_matrix[i][j]` is the number of residues deleted from
                the aligned sequence i at residue position j.
    """
    sequences, _ = parse_fasta(a3m_string)
    deletion_matrix = []
    for msa_sequence in sequences:
        deletion_vec = []
        deletion_count = 0
        for j in msa_sequence:
            if j.islower():
                deletion_count += 1
            else:
                deletion_vec.append(deletion_count)
                deletion_count = 0
        deletion_matrix.append(deletion_vec)

    # Make the MSA matrix out of aligned (deletion-free) sequences.
    deletion_table = str.maketrans("", "", string.ascii_lowercase)
    aligned_sequences = [s.translate(deletion_table) for s in sequences]
    return aligned_sequences, deletion_matrix


def _convert_sto_seq_to_a3m(
    query_non_gaps: Sequence[bool], sto_seq: str
) -> Iterable[str]:
    for is_query_res_non_gap, sequence_res in zip(query_non_gaps, sto_seq):
        if is_query_res_non_gap:
            yield sequence_res
        elif sequence_res != "-":
            yield sequence_res.lower()


def convert_stockholm_to_a3m(
    stockholm_format: str, max_sequences: Optional[int] = None
) -> str:
    """Converts MSA in Stockholm format to the A3M format."""
    descriptions = {}
    sequences = {}
    reached_max_sequences = False

    for line in stockholm_format.splitlines():
        reached_max_sequences = (
            max_sequences and len(sequences) >= max_sequences
        )
        if line.strip() and not line.startswith(("#", "//")):
            # Ignore blank lines, markup and end symbols - remainder are alignment
            # sequence parts.
            seqname, aligned_seq = line.split(maxsplit=1)
            if seqname not in sequences:
                if reached_max_sequences:
                    continue
                sequences[seqname] = ""
            sequences[seqname] += aligned_seq

    for line in stockholm_format.splitlines():
        if line[:4] == "#=GS":
            # Description row - example format is:
            # #=GS UniRef90_Q9H5Z4/4-78            DE [subseq from] cDNA: FLJ22755 ...
            columns = line.split(maxsplit=3)
            seqname, feature = columns[1:3]
            value = columns[3] if len(columns) == 4 else ""
            if feature != "DE":
                continue
            if reached_max_sequences and seqname not in sequences:
                continue
            descriptions[seqname] = value
            if len(descriptions) == len(sequences):
                break

    # Convert sto format to a3m line by line
    a3m_sequences = {}
    # query_sequence is assumed to be the first sequence
    query_sequence = next(iter(sequences.values()))
    query_non_gaps = [res != "-" for res in query_sequence]
    for seqname, sto_sequence in sequences.items():
        a3m_sequences[seqname] = "".join(
            _convert_sto_seq_to_a3m(query_non_gaps, sto_sequence)
        )

    fasta_chunks = (
        f">{k} {descriptions.get(k, '')}\n{a3m_sequences[k]}"
        for k in a3m_sequences
    )
    return "\n".join(fasta_chunks) + "\n"  # Include terminating newline.


def _get_hhr_line_regex_groups(
    regex_pattern: str, line: str
) -> Sequence[Optional[str]]:
    match = re.match(regex_pattern, line)
    if match is None:
        raise RuntimeError(f"Could not parse query line {line}")
    return match.groups()


def _update_hhr_residue_indices_list(
    sequence: str, start_index: int, indices_list: List[int]
):
    """Computes the relative indices for each residue with respect to the original sequence."""
    counter = start_index
    for symbol in sequence:
        if symbol == "-":
            indices_list.append(-1)
        else:
            indices_list.append(counter)
            counter += 1


def _parse_hhr_hit(detailed_lines: Sequence[str]) -> TemplateHit:
    """Parses the detailed HMM HMM comparison section for a single Hit.

    This works on .hhr files generated from both HHBlits and HHSearch.

    Args:
        detailed_lines: A list of lines from a single comparison section between 2
            sequences (which each have their own HMM's)

    Returns:
        A dictionary with the information from that detailed comparison section

    Raises:
        RuntimeError: If a certain line cannot be processed
    """
    # Parse first 2 lines.
    number_of_hit = int(detailed_lines[0].split()[-1])
    name_hit = detailed_lines[1][1:]

    # Parse the summary line.
    pattern = (
        "Probab=(.*)[\t ]*E-value=(.*)[\t ]*Score=(.*)[\t ]*Aligned_cols=(.*)[\t"
        " ]*Identities=(.*)%[\t ]*Similarity=(.*)[\t ]*Sum_probs=(.*)[\t "
        "]*Template_Neff=(.*)"
    )
    match = re.match(pattern, detailed_lines[2])
    if match is None:
        raise RuntimeError(
            "Could not parse section: %s. Expected this: \n%s to contain summary."
            % (detailed_lines, detailed_lines[2])
        )
    (prob_true, e_value, _, aligned_cols, _, _, sum_probs, neff) = [
        float(x) for x in match.groups()
    ]

    # The next section reads the detailed comparisons. These are in a 'human
    # readable' format which has a fixed length. The strategy employed is to
    # assume that each block starts with the query sequence line, and to parse
    # that with a regexp in order to deduce the fixed length used for that block.
    query = ""
    hit_sequence = ""
    indices_query = []
    indices_hit = []
    length_block = None

    for line in detailed_lines[3:]:
        # Parse the query sequence line
        if (
            line.startswith("Q ")
            and not line.startswith("Q ss_dssp")
            and not line.startswith("Q ss_pred")
            and not line.startswith("Q Consensus")
        ):
            # Thus the first 17 characters must be 'Q <query_name> ', and we can parse
            # everything after that.
            #              start    sequence       end       total_sequence_length
            patt = r"[\t ]*([0-9]*) ([A-Z-]*)[\t ]*([0-9]*) \([0-9]*\)"
            groups = _get_hhr_line_regex_groups(patt, line[17:])

            # Get the length of the parsed block using the start and finish indices,
            # and ensure it is the same as the actual block length.
            start = int(groups[0]) - 1  # Make index zero based.
            delta_query = groups[1]
            end = int(groups[2])
            num_insertions = len([x for x in delta_query if x == "-"])
            length_block = end - start + num_insertions
            assert length_block == len(delta_query)

            # Update the query sequence and indices list.
            query += delta_query
            _update_hhr_residue_indices_list(delta_query, start, indices_query)

        elif line.startswith("T "):
            # Parse the hit sequence.
            if (
                not line.startswith("T ss_dssp")
                and not line.startswith("T ss_pred")
                and not line.startswith("T Consensus")
            ):
                # Thus the first 17 characters must be 'T <hit_name> ', and we can
                # parse everything after that.
                #              start    sequence       end     total_sequence_length
                patt = r"[\t ]*([0-9]*) ([A-Z-]*)[\t ]*[0-9]* \([0-9]*\)"
                groups = _get_hhr_line_regex_groups(patt, line[17:])
                start = int(groups[0]) - 1  # Make index zero based.
                delta_hit_sequence = groups[1]
                assert length_block == len(delta_hit_sequence)

                # Update the hit sequence and indices list.
                hit_sequence += delta_hit_sequence
                _update_hhr_residue_indices_list(
                    delta_hit_sequence, start, indices_hit
                )

    return TemplateHit(
        index=number_of_hit,
        name=name_hit,
        aligned_cols=int(aligned_cols),
        sum_probs=sum_probs,
        query=query,
        hit_sequence=hit_sequence,
        indices_query=indices_query,
        indices_hit=indices_hit,
    )


def parse_hhr(hhr_string: str) -> Sequence[TemplateHit]:
    """Parses the content of an entire HHR file."""
    lines = hhr_string.splitlines()

    # Each .hhr file starts with a results table, then has a sequence of hit
    # "paragraphs", each paragraph starting with a line 'No <hit number>'. We
    # iterate through each paragraph to parse each hit.

    block_starts = [i for i, line in enumerate(lines) if line.startswith("No ")]

    hits = []
    if block_starts:
        block_starts.append(len(lines))  # Add the end of the final block.
        for i in range(len(block_starts) - 1):
            hits.append(
                _parse_hhr_hit(lines[block_starts[i] : block_starts[i + 1]])
            )
    return hits


def parse_e_values_from_tblout(tblout: str) -> Dict[str, float]:
    """Parse target to e-value mapping parsed from Jackhmmer tblout string."""
    e_values = {"query": 0}
    lines = [line for line in tblout.splitlines() if line[0] != "#"]
    # As per http://eddylab.org/software/hmmer/Userguide.pdf fields are
    # space-delimited. Relevant fields are (1) target name:  and
    # (5) E-value (full sequence) (numbering from 1).
    for line in lines:
        fields = line.split()
        e_value = fields[4]
        target_name = fields[0]
        e_values[target_name] = float(e_value)
    return e_values



@contextlib.contextmanager
def tmpdir_manager(base_dir: Optional[str] = None):
    """Context manager that deletes a temporary directory on exit."""
    tmpdir = tempfile.mkdtemp(dir=base_dir)
    try:
        yield tmpdir
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@contextlib.contextmanager
def timing(msg: str):
    logging.info("Started %s", msg)
    tic = time.perf_counter()
    yield
    toc = time.perf_counter()
    logging.info("Finished %s in %.3f seconds", msg, toc - tic)


def to_date(s: str):
    return datetime.datetime(
        year=int(s[:4]), month=int(s[5:7]), day=int(s[8:10])
    )



_HHBLITS_DEFAULT_P = 20
_HHBLITS_DEFAULT_Z = 500


class HHBlits:
    """Python wrapper of the HHblits binary."""

    def __init__(
        self,
        *,
        binary_path: str,
        databases: Sequence[str],
        n_cpu: int = 4,
        n_iter: int = 3,
        e_value: float = 0.001,
        maxseq: int = 1_000_000,
        realign_max: int = 100_000,
        maxfilt: int = 100_000,
        min_prefilter_hits: int = 1000,
        all_seqs: bool = False,
        alt: Optional[int] = None,
        p: int = _HHBLITS_DEFAULT_P,
        z: int = _HHBLITS_DEFAULT_Z,
    ):
        """Initializes the Python HHblits wrapper.

        Args:
          binary_path: The path to the HHblits executable.
          databases: A sequence of HHblits database paths. This should be the
            common prefix for the database files (i.e. up to but not including
            _hhm.ffindex etc.)
          n_cpu: The number of CPUs to give HHblits.
          n_iter: The number of HHblits iterations.
          e_value: The E-value, see HHblits docs for more details.
          maxseq: The maximum number of rows in an input alignment. Note that this
            parameter is only supported in HHBlits version 3.1 and higher.
          realign_max: Max number of HMM-HMM hits to realign. HHblits default: 500.
          maxfilt: Max number of hits allowed to pass the 2nd prefilter.
            HHblits default: 20000.
          min_prefilter_hits: Min number of hits to pass prefilter.
            HHblits default: 100.
          all_seqs: Return all sequences in the MSA / Do not filter the result MSA.
            HHblits default: False.
          alt: Show up to this many alternative alignments.
          p: Minimum Prob for a hit to be included in the output hhr file.
            HHblits default: 20.
          z: Hard cap on number of hits reported in the hhr file.
            HHblits default: 500. NB: The relevant HHblits flag is -Z not -z.

        Raises:
          RuntimeError: If HHblits binary not found within the path.
        """
        self.binary_path = binary_path
        self.databases = databases

        for database_path in self.databases:
            if not glob.glob(database_path + "_*"):
                logging.error(
                    "Could not find HHBlits database %s", database_path
                )
                raise ValueError(
                    f"Could not find HHBlits database {database_path}"
                )

        self.n_cpu = n_cpu
        self.n_iter = n_iter
        self.e_value = e_value
        self.maxseq = maxseq
        self.realign_max = realign_max
        self.maxfilt = maxfilt
        self.min_prefilter_hits = min_prefilter_hits
        self.all_seqs = all_seqs
        self.alt = alt
        self.p = p
        self.z = z

    def query(self, input_fasta_path: str) -> Mapping[str, Any]:
        """Queries the database using HHblits."""
        with tmpdir_manager(base_dir="/tmp") as query_tmp_dir:
            a3m_path = os.path.join(query_tmp_dir, "output.a3m")

            db_cmd = []
            for db_path in self.databases:
                db_cmd.append("-d")
                db_cmd.append(db_path)
            cmd = [
                self.binary_path,
                "-i",
                input_fasta_path,
                "-cpu",
                str(self.n_cpu),
                "-oa3m",
                a3m_path,
                "-o",
                "/dev/null",
                "-n",
                str(self.n_iter),
                "-e",
                str(self.e_value),
                "-maxseq",
                str(self.maxseq),
                "-realign_max",
                str(self.realign_max),
                "-maxfilt",
                str(self.maxfilt),
                "-min_prefilter_hits",
                str(self.min_prefilter_hits),
            ]
            if self.all_seqs:
                cmd += ["-all"]
            if self.alt:
                cmd += ["-alt", str(self.alt)]
            if self.p != _HHBLITS_DEFAULT_P:
                cmd += ["-p", str(self.p)]
            if self.z != _HHBLITS_DEFAULT_Z:
                cmd += ["-Z", str(self.z)]
            cmd += db_cmd

            logging.info('Launching subprocess "%s"', " ".join(cmd))
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )

            with timing("HHblits query"):
                stdout, stderr = process.communicate()
                retcode = process.wait()

            if retcode:
                # Logs have a 15k character limit, so log HHblits error line by line.
                logging.error("HHblits failed. HHblits stderr begin:")
                for error_line in stderr.decode("utf-8").splitlines():
                    if error_line.strip():
                        logging.error(error_line.strip())
                logging.error("HHblits stderr end")
                raise RuntimeError(
                    "HHblits failed\nstdout:\n%s\n\nstderr:\n%s\n"
                    % (stdout.decode("utf-8"), stderr[:500_000].decode("utf-8"))
                )

            with open(a3m_path) as f:
                a3m = f.read()

        raw_output = dict(
            a3m=a3m,
            output=stdout,
            stderr=stderr,
            n_iter=self.n_iter,
            e_value=self.e_value,
        )
        return raw_output

class Jackhmmer:
    """Python wrapper of the Jackhmmer binary."""

    def __init__(
        self,
        *,
        binary_path: str,
        database_path: str,
        n_cpu: int = 8,
        n_iter: int = 1,
        e_value: float = 0.0001,
        z_value: Optional[int] = None,
        get_tblout: bool = False,
        filter_f1: float = 0.0005,
        filter_f2: float = 0.00005,
        filter_f3: float = 0.0000005,
        incdom_e: Optional[float] = None,
        dom_e: Optional[float] = None,
        num_streamed_chunks: Optional[int] = None,
        streaming_callback: Optional[Callable[[int], None]] = None,
    ):
        """Initializes the Python Jackhmmer wrapper.

        Args:
          binary_path: The path to the jackhmmer executable.
          database_path: The path to the jackhmmer database (FASTA format).
          n_cpu: The number of CPUs to give Jackhmmer.
          n_iter: The number of Jackhmmer iterations.
          e_value: The E-value, see Jackhmmer docs for more details.
          z_value: The Z-value, see Jackhmmer docs for more details.
          get_tblout: Whether to save tblout string.
          filter_f1: MSV and biased composition pre-filter, set to >1.0 to turn off.
          filter_f2: Viterbi pre-filter, set to >1.0 to turn off.
          filter_f3: Forward pre-filter, set to >1.0 to turn off.
          incdom_e: Domain e-value criteria for inclusion of domains in MSA/next
            round.
          dom_e: Domain e-value criteria for inclusion in tblout.
          num_streamed_chunks: Number of database chunks to stream over.
          streaming_callback: Callback function run after each chunk iteration with
            the iteration number as argument.
        """
        self.binary_path = binary_path
        self.database_path = database_path
        self.num_streamed_chunks = num_streamed_chunks

        if (
            not os.path.exists(self.database_path)
            and num_streamed_chunks is None
        ):
            logging.error("Could not find Jackhmmer database %s", database_path)
            raise ValueError(
                f"Could not find Jackhmmer database {database_path}"
            )

        self.n_cpu = n_cpu
        self.n_iter = n_iter
        self.e_value = e_value
        self.z_value = z_value
        self.filter_f1 = filter_f1
        self.filter_f2 = filter_f2
        self.filter_f3 = filter_f3
        self.incdom_e = incdom_e
        self.dom_e = dom_e
        self.get_tblout = get_tblout
        self.streaming_callback = streaming_callback

    def _query_chunk(
        self, input_fasta_path: str, database_path: str
    ) -> Mapping[str, Any]:
        """Queries the database chunk using Jackhmmer."""
        with tmpdir_manager(base_dir="/tmp") as query_tmp_dir:
            sto_path = os.path.join(query_tmp_dir, "output.sto")

            # The F1/F2/F3 are the expected proportion to pass each of the filtering
            # stages (which get progressively more expensive), reducing these
            # speeds up the pipeline at the expensive of sensitivity.  They are
            # currently set very low to make querying Mgnify run in a reasonable
            # amount of time.
            cmd_flags = [
                # Don't pollute stdout with Jackhmmer output.
                "-o",
                "/dev/null",
                "-A",
                sto_path,
                "--noali",
                "--F1",
                str(self.filter_f1),
                "--F2",
                str(self.filter_f2),
                "--F3",
                str(self.filter_f3),
                "--incE",
                str(self.e_value),
                # Report only sequences with E-values <= x in per-sequence output.
                "-E",
                str(self.e_value),
                "--cpu",
                str(self.n_cpu),
                "-N",
                str(self.n_iter),
            ]
            if self.get_tblout:
                tblout_path = os.path.join(query_tmp_dir, "tblout.txt")
                cmd_flags.extend(["--tblout", tblout_path])

            if self.z_value:
                cmd_flags.extend(["-Z", str(self.z_value)])

            if self.dom_e is not None:
                cmd_flags.extend(["--domE", str(self.dom_e)])

            if self.incdom_e is not None:
                cmd_flags.extend(["--incdomE", str(self.incdom_e)])

            cmd = (
                [self.binary_path]
                + cmd_flags
                + [input_fasta_path, database_path]
            )

            logging.info('Launching subprocess "%s"', " ".join(cmd))
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            with timing(
                f"Jackhmmer ({os.path.basename(database_path)}) query"
            ):
                _, stderr = process.communicate()
                retcode = process.wait()

            if retcode:
                raise RuntimeError(
                    "Jackhmmer failed\nstderr:\n%s\n" % stderr.decode("utf-8")
                )

            # Get e-values for each target name
            tbl = ""
            if self.get_tblout:
                with open(tblout_path) as f:
                    tbl = f.read()

            with open(sto_path) as f:
                sto = f.read()

        raw_output = dict(
            sto=sto,
            tbl=tbl,
            stderr=stderr,
            n_iter=self.n_iter,
            e_value=self.e_value,
        )

        return raw_output

    def query(self, input_fasta_path: str) -> Sequence[Mapping[str, Any]]:
        """Queries the database using Jackhmmer."""
        if self.num_streamed_chunks is None:
            return [self._query_chunk(input_fasta_path, self.database_path)]

        db_basename = os.path.basename(self.database_path)
        db_remote_chunk = lambda db_idx: f"{self.database_path}.{db_idx}"
        db_local_chunk = lambda db_idx: f"/tmp/ramdisk/{db_basename}.{db_idx}"

        # Remove existing files to prevent OOM
        for f in glob.glob(db_local_chunk("[0-9]*")):
            try:
                os.remove(f)
            except OSError:
                print(f"OSError while deleting {f}")

        # Download the (i+1)-th chunk while Jackhmmer is running on the i-th chunk
        with futures.ThreadPoolExecutor(max_workers=2) as executor:
            chunked_output = []
            for i in range(1, self.num_streamed_chunks + 1):
                # Copy the chunk locally
                if i == 1:
                    future = executor.submit(
                        request.urlretrieve,
                        db_remote_chunk(i),
                        db_local_chunk(i),
                    )
                if i < self.num_streamed_chunks:
                    next_future = executor.submit(
                        request.urlretrieve,
                        db_remote_chunk(i + 1),
                        db_local_chunk(i + 1),
                    )

                # Run Jackhmmer with the chunk
                future.result()
                chunked_output.append(
                    self._query_chunk(input_fasta_path, db_local_chunk(i))
                )

                # Remove the local copy of the chunk
                os.remove(db_local_chunk(i))
                future = next_future
                if self.streaming_callback:
                    self.streaming_callback(i)
        return chunked_output


class AlignmentRunner:
    """Runs alignment tools and saves the results"""
    def __init__(
        self,
        jackhmmer_binary_path ='.../hmmer-3.3.2/src/jackhmmer',
        hhblits_binary_path = '.../hh-suite/build/bin/hhblits',
        uniref90_database_path = '.../uniref90_2020_0422/uniref90.fasta',
        mgnify_database_path = '.../mgnify_2019_06/mgnify.fasta',
        bfd_database_path = '.../bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt',
        uniclust30_database_path = '.../UniRef30_2020_03/UniRef30_2020_03',
        no_cpus = 8,
        uniref_max_hits: int = 10000,
        mgnify_max_hits: int = 5000,
    ):
        self.uniref_max_hits = uniref_max_hits
        self.mgnify_max_hits = mgnify_max_hits


        self.jackhmmer_uniref90_runner = Jackhmmer(
            binary_path=jackhmmer_binary_path,
            database_path=uniref90_database_path,
            n_cpu=no_cpus,
        )
   
        dbs = [bfd_database_path, uniclust30_database_path]
        self.hhblits_bfd_uniclust_runner = HHBlits(
            binary_path=hhblits_binary_path,
            databases=dbs,
            n_cpu=no_cpus,
        )

        self.jackhmmer_mgnify_runner = Jackhmmer(
            binary_path=jackhmmer_binary_path,
            database_path=mgnify_database_path,
            n_cpu=no_cpus,
        )

    def run(self,fasta_path: str,output_dir: str,):
        hhblits_bfd_uniclust_result = (self.hhblits_bfd_uniclust_runner.query(fasta_path))
        bfd_out_path = os.path.join(output_dir, "bfd_uniclust_hits.a3m")
        with open(bfd_out_path, "w") as f:
            f.write(hhblits_bfd_uniclust_result["a3m"])
                    
        jackhmmer_uniref90_result = self.jackhmmer_uniref90_runner.query(fasta_path)[0]
        uniref90_msa_as_a3m = convert_stockholm_to_a3m(
            jackhmmer_uniref90_result["sto"], 
            max_sequences=self.uniref_max_hits
        )
        uniref90_out_path = os.path.join(output_dir, "uniref90_hits.a3m")
        with open(uniref90_out_path, "w") as f:
            f.write(uniref90_msa_as_a3m)

        jackhmmer_mgnify_result = self.jackhmmer_mgnify_runner.query(fasta_path)[0]
        mgnify_msa_as_a3m = convert_stockholm_to_a3m(
            jackhmmer_mgnify_result["sto"], 
            max_sequences=self.mgnify_max_hits
        )
        mgnify_out_path = os.path.join(output_dir, "mgnify_hits.a3m")
        with open(mgnify_out_path, "w") as f:
            f.write(mgnify_msa_as_a3m)

MSARunner = AlignmentRunner()
MSARunner.run(f'example/target.fasta', f'example/')

with open(f'example/test_1.a3m', 'w') as fout:
    for file in ['uniref90_hits', 'bfd_uniclust_hits.a3m', 'mgnify_hits.a3m']:
        with open(f'example/{file}.a3m') as fin:
            for line in fin.readlines():
                if line[0] != '>':
                    for AA in list(line.strip()):
                        if ord(AA) < 97:
                            fout.write(AA)
                        else:
                            fout.write('-')
                    fout.write('\n')
                else:
                    fout.write(line)
