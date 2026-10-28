# -*- coding: utf-8 -*-
"""
QR Code Generator Module

This module provides the core functionality for generating QR codes with advanced
configuration options. Originally developed for Yape QR regeneration, it has
evolved into a comprehensive QR code generation tool.

Functions:
    make_qr: Generate QR code with specified parameters
    evaluate_all_masks: Evaluate all mask patterns to find optimal one
"""

import re
import segno
from segno import consts as _c
from typing import List, Optional, Union, Tuple, Dict, Any
from .penalties import compute_mask_penalty

# ---------------------------------------------------------------------------
# Mixed-mode (multi-segment) encoding
#
# A QR symbol can switch between Numeric / Alphanumeric / Byte several times
# inside the same message. EMVCo payloads (Yape, Plin, Lemon, Dale, ...) are
# long runs of digits with a few lowercase letters, so encoders such as the
# ones used by those apps split the text into segments and fit it in a much
# smaller version than a single Byte segment would need.
# segno only does this when it is given an explicit list of segments, so
# mode='mixed' below computes the optimal split (dynamic programming) and
# hands the segments to segno.
# ---------------------------------------------------------------------------
MIXED_MODE = 'mixed'

_ALNUM_CHARS = set('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:')
_NUM, _ALN, _BYT = _c.MODE_NUMERIC, _c.MODE_ALPHANUMERIC, _c.MODE_BYTE


def _count_bits(version: int) -> Dict[int, int]:
    """Bits of the character-count indicator per mode for a QR version."""
    if version <= 9:
        return {_NUM: 10, _ALN: 9, _BYT: 8}
    if version <= 26:
        return {_NUM: 12, _ALN: 11, _BYT: 16}
    return {_NUM: 14, _ALN: 13, _BYT: 16}


def _segment_bits(mode: int, n: int, nbytes: int, cc: Dict[int, int]) -> int:
    """Bit cost of one segment (mode indicator + count + payload)."""
    if mode == _NUM:
        return 4 + cc[_NUM] + 10 * (n // 3) + (0, 4, 7)[n % 3]
    if mode == _ALN:
        return 4 + cc[_ALN] + 11 * (n // 2) + 6 * (n % 2)
    return 4 + cc[_BYT] + 8 * nbytes


# Data codewords per (version, ecc): used only to pick the smallest version.
def _data_capacity_bits(version: int, ecc: str) -> int:
    level = {'L': _c.ERROR_LEVEL_L, 'M': _c.ERROR_LEVEL_M,
             'Q': _c.ERROR_LEVEL_Q, 'H': _c.ERROR_LEVEL_H}[ecc.upper()]
    return sum(e.num_blocks * e.num_data for e in _c.ECC[version][level]) * 8


def optimal_segments(text: str, version: int, encoding: str = 'utf-8',
                     ) -> Tuple[List[Tuple[str, int, str]], int]:
    """
    Split ``text`` into the Numeric/Alphanumeric/Byte segments that need the
    fewest bits for the given version.

    Returns ``([(chunk, segno_mode, encoding), ...], total_bits)`` where
    ``total_bits`` excludes the ECI header and the terminator.
    """
    cc = _count_bits(version)
    n = len(text)
    INF = 10 ** 9
    best = [(INF, None)] * (n + 1)
    best[0] = (0, None)
    blen = [len(ch.encode(encoding)) for ch in text]

    for i in range(n):
        if best[i][0] >= INF:
            continue
        for mode, ok in ((_NUM, str.isdigit),
                         (_ALN, lambda ch: ch in _ALNUM_CHARS),
                         (_BYT, lambda ch: True)):
            nbytes = 0
            for j in range(i + 1, n + 1):
                ch = text[j - 1]
                if mode == _NUM and not (ch.isascii() and ch.isdigit()):
                    break
                if not ok(ch):
                    break
                nbytes += blen[j - 1]
                cost = best[i][0] + _segment_bits(mode, j - i, nbytes, cc)
                if cost < best[j][0]:
                    best[j] = (cost, (i, mode))

    segs, j = [], n
    while j > 0:
        i, mode = best[j][1]
        segs.append((text[i:j], mode, encoding))
        j = i
    segs.reverse()
    return segs, best[n][0]


def _mixed_segments(text: str, ecc: str, version: Optional[int],
                    encoding: str, eci: bool):
    """Return (segments, version) for mode='mixed', choosing the smallest
    version that fits when ``version`` is None."""
    eci_bits = 12 if eci else 0
    versions = [version] if version else range(1, 41)
    for v in versions:
        segs, bits = optimal_segments(text, v, encoding)
        if bits + eci_bits <= _data_capacity_bits(v, ecc):
            return segs, v
    raise ValueError('The data does not fit in any QR version with '
                     f'ecc={ecc}' if not version else
                     f'The data does not fit into version {version} with '
                     f'mixed segments and ecc={ecc}')


def make_qr(
    text: str,
    ecc: str = 'M',
    version: Optional[Union[int, str]] = None,
    mode: str = 'byte',
    encoding: str = 'utf-8',
    eci: bool = True,
    mask: Union[str, int] = 'auto',
    boost_error: bool = False,
    micro: bool = False
) -> segno.QRCode:
    """
    Generate a QR code symbol with specified parameters.
    
    This function creates a QR code using the segno library with advanced
    configuration options. Originally designed for Yape QR regeneration,
    it supports all standard QR code parameters.
    
    Args:
        text (str): The data to encode in the QR code
        ecc (str): Error correction level ('L', 'M', 'Q', 'H')
            - L: ~7% recovery capability
            - M: ~15% recovery capability (Yape default)
            - Q: ~25% recovery capability  
            - H: ~30% recovery capability
        version (Optional[Union[int, str]]): QR code version (1-40) or 'auto'
            - 'auto': Select minimum version that fits the data
            - int: Force specific version (1=21x21, 40=177x177)
        mode (str): Encoding mode
            - 'byte': Any binary data/UTF-8 (recommended, Yape uses this)
            - 'alphanumeric': A-Z, 0-9, and few symbols (more compact)
            - 'numeric': Digits only (maximum compression)
            - 'kanji': Shift-JIS characters
            - 'mixed': optimal split into numeric/alphanumeric/byte segments
              (what Yape/Plin/Lemon/Dale QR codes use; fits EMVCo payloads in
              a much smaller version)
        encoding (str): Character encoding for byte mode (e.g., 'utf-8', 'iso-8859-1')
        eci (bool): Extended Channel Interpretation
            - True: Add ECI header with encoding info (e.g., UTF-8)
            - False: Assume ISO-8859-1 implicitly
        mask (Union[str, int]): Mask pattern
            - 'auto': Calculate optimal mask using ISO/IEC penalty rules
            - int: Use specific mask pattern (0-7, Yape uses 2)
        boost_error (bool): Automatically increase ECC level if space allows
        micro (bool): Use Micro QR format (M1-M4) instead of standard QR
        
    Returns:
        segno.QRCode: Generated QR code object
        
    Raises:
        ValueError: If parameters are invalid
        segno.DataOverflowError: If data doesn't fit in specified version
        
    Example:
        >>> # Generate Yape-style QR code
        >>> qr = make_qr("00020101021243650016COM.MERCADOLIVRE02008...", 
        ...              ecc='M', mask=2, mode='byte')
        >>> 
        >>> # Generate with auto-optimization
        >>> qr = make_qr("https://example.com", ecc='M', version='auto', mask='auto')
    """
    # Convert mask parameter: 'auto' -> None, otherwise int
    mask_arg = None if mask == 'auto' else int(mask)
    
    # Convert version parameter: 'auto' or None -> None, otherwise int
    ver_arg = None if (version in (None, 'auto')) else int(version)
    
    if (mode or '').lower() == MIXED_MODE:
        if micro:
            raise ValueError("mode='mixed' is not available for Micro QR codes")
        segments, ver_arg = _mixed_segments(text, ecc, ver_arg, encoding, bool(eci))
        return segno.make_qr(
            segments,
            error=ecc,
            version=ver_arg,
            mask=mask_arg,
            eci=bool(eci),
            boost_error=bool(boost_error)
        )

    return segno.make(
        text,
        error=ecc,
        version=ver_arg,
        mode=mode,
        encoding=encoding,
        eci=bool(eci),
        mask=mask_arg,
        boost_error=bool(boost_error),
        micro=bool(micro)
    )


def evaluate_all_masks(
    text: str,
    ecc: str,
    version: int,
    mode: str,
    encoding: str,
    eci: bool,
    boost_error: bool,
    micro: bool
) -> Tuple[int, int, Dict[int, int]]:
    """
    Evaluate all mask patterns (0-7) to find the optimal one.
    
    This function generates QR codes with all possible mask patterns and
    calculates their penalty scores according to ISO/IEC 18004 standard.
    The mask with the lowest penalty score is considered optimal.
    
    Args:
        text (str): The data to encode
        ecc (str): Error correction level ('L', 'M', 'Q', 'H')
        version (int): QR code version (1-40)
        mode (str): Encoding mode ('byte', 'alphanumeric', 'numeric', 'kanji')
        encoding (str): Character encoding for byte mode
        eci (bool): Extended Channel Interpretation flag
        boost_error (bool): Boost error correction flag
        micro (bool): Micro QR format flag
        
    Returns:
        Tuple[int, int, Dict[int, int]]: (best_mask, best_score, all_scores)
            - best_mask: Mask pattern with lowest penalty (0-7)
            - best_score: Penalty score of the best mask
            - all_scores: Dictionary mapping mask -> penalty score
            
    Example:
        >>> best_mask, best_score, scores = evaluate_all_masks(
        ...     "Hello World", ecc='M', version=2, mode='byte',
        ...     encoding='utf-8', eci=True, boost_error=False, micro=False
        ... )
        >>> print(f"Best mask: {best_mask} (score: {best_score})")
        >>> print(f"All scores: {scores}")
    """
    scores = {}
    best_mask = None
    best_score = None
    
    # Evaluate each mask pattern (0-7)
    for mask_pattern in range(8):
        try:
            # Generate QR code with specific mask
            symbol = make_qr(
                text, ecc=ecc, version=version, mode=mode,
                encoding=encoding, eci=eci, mask=mask_pattern,
                boost_error=boost_error, micro=micro
            )
            
            # Convert matrix to list format for penalty calculation
            matrix = list(symbol.matrix)
            penalty_score = compute_mask_penalty(matrix)
            
            scores[mask_pattern] = penalty_score
            
            # Track the best (lowest penalty) mask
            if best_score is None or penalty_score < best_score:
                best_score = penalty_score
                best_mask = mask_pattern
                
        except Exception:
            # If a mask fails, assign high penalty
            scores[mask_pattern] = 999999
    
    return best_mask, best_score, scores
