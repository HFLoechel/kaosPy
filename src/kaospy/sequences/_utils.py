import warnings

from kaospy.sequences.symbol_sets import predefined_alphabets


def normalize_sequence(sequence: str | int | list[str | int]) -> list[str]:
    """
    Normalize the input sequence into a list of strings.

    For string and integer inputs, the sequence is converted into a list
    of individual string representations. Lists containing strings or
    integers are converted element-wise to strings.

    :param sequence: Input sequence; can be a string, an integer, or a list of strings/integers.
    :type sequence: str | int | list[str | int]
    :return: Normalized sequence as a list of strings.
    :type: list[str]
    """

    if isinstance(sequence, list):
        for x in sequence:
            if not isinstance(x, (str, int)):
                raise TypeError("Elements of data must be strings or integers.")
        data_list = [str(x) for x in sequence]

    else:
        if not isinstance(sequence, (str, int)):
            raise TypeError("data must be a string or integer, or a list of these.")
        data_list = list(str(sequence))
    return data_list


def build_symbols(sequence: list[str], symbols: list[str] | str | None = None) -> list[str]:
    """
    Return the set of symbols used for the given sequence.

    If symbols are provided, they are validated and used directly.
    The symbols parameter can either be a user-defined list of symbols
    or a key referring to a predefined symbol set.

    If no symbols are provided, they are generated automatically from
    the input sequence.

    :param sequence: Normalized input sequence as a list of symbols.
    :type sequence: list[str]
    :param symbols: Optional user-defined symbols or key referring to a
                    predefined symbol set.
    :type symbols: list[str] | str | None
    :return: List of symbols used for the CGR representation.
    :rtype: list[str]
    """

    if symbols is not None:
        # symbols provided as a list
        if isinstance(symbols, list):
            if len(symbols) != len(set(symbols)):
                raise ValueError("The given symbols contain duplicate entries.")

            if not all(isinstance(x, str) for x in symbols):
                raise TypeError("All symbols must be strings.")

            final_symbols = symbols

        # symbols provided as predefined key
        elif isinstance(symbols, str):
            try:
                final_symbols = predefined_alphabets[symbols]
            except KeyError:
                raise ValueError(
                    f"'{symbols}' is not a predefined symbol set key. "
                    f"Choose one of: {', '.join(predefined_alphabets.keys())}"
                )

        else:
            raise TypeError(
                "symbols must be a list of strings or a string key "
                "for a predefined symbol set."
            )

        # check that all sequence values are represented by the symbols
        invalid = set(sequence) - set(final_symbols)

        if invalid:
            raise ValueError(
                f"Sequence contains values not present in the symbols.\n"
                f"Invalid values: {sorted(invalid)}\n"
                f"Symbols: {sorted(final_symbols)}"
            )

    # no symbols provided → generate automatically
    else:
        final_symbols = sorted(set(sequence))
        warnings.warn(
            f"No symbols provided: automatically generated symbols = {final_symbols}",
             stacklevel = 3
        )

    return final_symbols
