import re
from statistics import median


def looks_like_program_code(text):
    """
    Return True when an OCR line strongly resembles source code.

    The function intentionally uses several small signals instead of
    relying on one keyword, because OCR may slightly damage code text.
    """
    text = str(text).strip()

    if not text:
        return False

    # A plain number near the bottom of a page is more likely a page
    # number. Page-number classification is also checked separately.
    if re.fullmatch(r"\d{1,4}", text):
        return False

    score = 0

    programming_keywords = re.compile(
        r"^\s*("
        r"def|class|return|import|from|for|while|"
        r"if|elif|else|try|except|finally|with|"
        r"lambda|print|assert|yield|raise|pass|"
        r"break|continue|async|await|"
        r"public|private|protected|static|interface|"
        r"void|int|float|double|boolean|char|String|"
        r"package|new|switch|case|catch|throws"
        r")\b"
    )

    if programming_keywords.match(text):
        score += 3

    # Variable assignment, for example:
    # encoder = ...
    # gen_code_keys = ...
    if re.search(
        r"\b[A-Za-z_][A-Za-z0-9_]*\s*=\s*(?!=)",
        text
    ):
        score += 3

    if re.search(r"\blambda\b", text):
        score += 3

    # Snake-case identifiers.
    if re.search(
        r"[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+",
        text
    ):
        score += 1

    # Function or method call near the beginning of a line.
    if re.match(
        r"^[A-Za-z_][A-Za-z0-9_]*"
        r"(?:\.[A-Za-z_][A-Za-z0-9_]*)*"
        r"\s*\(",
        text
    ):
        score += 2

    # Lists, dictionaries, sets and indexing.
    if re.search(r"[\[\]{}]", text):
        score += 2

    # Common programming operators.
    if re.search(
        r"==|!=|<=|>=|:=|\+=|-=|\*=|/=|//=|\*\*=|=>|->",
        text
    ):
        score += 2

    # Code continuation lines often start or end with punctuation.
    if re.match(r"^[\)\]\}\.,]", text):
        score += 2

    if re.search(r"[\(\[\{,:\\]$", text):
        score += 2

    symbol_count = sum(
        character in "()[]{}:=,+-*/.%'\"\\"
        for character in text
    )

    symbol_ratio = (
        symbol_count / max(len(text), 1)
    )

    if symbol_ratio >= 0.10:
        score += 1

    # Program output such as:
    # 1*13*2*6*57*2*1*13
    if (
        re.fullmatch(
            r"[\d\s*+\-/.]+",
            text
        )
        and any(
            operator in text
            for operator in "*+-/"
        )
    ):
        score += 3

    return score >= 3


def classify_text_blocks(
    ocr_results,
    page_height=None,
    page_width=None
):
    """
    Classify OCR regions as paragraph, heading, subheading,
    code or page number.
    """
    if not ocr_results:
        return []

    sorted_results = sorted(
        ocr_results,
        key=lambda item: (
            item.get("top", 0),
            item.get("left", 0)
        )
    )

    valid_heights = [
        item.get("height", 0)
        for item in sorted_results
        if item.get("height", 0) > 0
    ]

    typical_height = (
        median(valid_heights)
        if valid_heights
        else 20
    )

    if page_height is None:
        page_height = max(
            item.get("bottom", 0)
            for item in sorted_results
        )

    if page_width is None:
        page_width = max(
            item.get("right", 0)
            for item in sorted_results
        )

    page_height = max(page_height, 1)
    page_width = max(page_width, 1)
    page_center = page_width / 2

    code_pattern = re.compile(
        r"^\s*("
        r"public|private|protected|static|class|interface|"
        r"void|int|float|double|boolean|char|String|"
        r"return|import|package|new|if|elif|else|for|while|"
        r"try|except|catch|finally|throws|System\.|"
        r"def|lambda|print|from|with|"
        r"//|/\*|\*|\{|\}"
        r")\b"
    )

    small_title_words = {
        "a",
        "an",
        "and",
        "as",
        "at",
        "by",
        "for",
        "from",
        "in",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with"
    }

    classified_blocks = []

    for index, item in enumerate(sorted_results):
        text = str(
            item.get("text", "")
        ).strip()

        if not text:
            continue

        word_count = len(
            text.split()
        )

        previous_item = (
            sorted_results[index - 1]
            if index > 0
            else None
        )

        next_item = (
            sorted_results[index + 1]
            if index + 1 < len(sorted_results)
            else None
        )

        gap_before = 0
        gap_after = 0

        if previous_item is not None:
            gap_before = max(
                0,
                item.get("top", 0)
                - previous_item.get("bottom", 0)
            )

        if next_item is not None:
            gap_after = max(
                0,
                next_item.get("top", 0)
                - item.get("bottom", 0)
            )

        left = item.get("left", 0)
        right = item.get("right", 0)
        top = item.get("top", 0)
        bottom = item.get("bottom", 0)
        height = item.get("height", 0)

        center_x = (
            left + right
        ) / 2

        centered = (
            abs(center_x - page_center)
            <= page_width * 0.16
        )

        near_page_top = (
            top < page_height * 0.25
        )

        near_page_bottom = (
            bottom > page_height * 0.86
        )

        noticeably_larger = (
            height >= typical_height * 1.12
        )

        strongly_larger = (
            height >= typical_height * 1.35
        )

        separated_from_body = (
            gap_before >= typical_height * 0.75
            or gap_after >= typical_height * 0.55
        )

        alphabetic_words = re.findall(
            r"[A-Za-z][A-Za-z'-]*",
            text
        )

        important_words = [
            word
            for word in alphabetic_words
            if word.lower() not in small_title_words
        ]

        is_title_case = bool(
            important_words
        ) and all(
            word[0].isupper()
            for word in important_words
            if word
        )

        has_letters = any(
            character.isalpha()
            for character in text
        )

        is_uppercase_title = (
            has_letters
            and text.upper() == text
        )

        ends_like_sentence = text.endswith(
            (
                ".",
                "?",
                "!",
                ",",
                ";",
                ":"
            )
        )

        short_heading_candidate = (
            1 <= word_count <= 9
            and len(text) <= 95
            and not ends_like_sentence
            and "(" not in text
            and ")" not in text
        )

        numeric_page_number = bool(
            re.fullmatch(
                r"\d{1,4}",
                text
            )
        )

        roman_page_number = bool(
            re.fullmatch(
                r"[ivxlcdm]+",
                text,
                flags=re.IGNORECASE
            )
        )

        looks_like_code = (
            bool(code_pattern.match(text))
            or looks_like_program_code(text)
        )

        # Parenthesized short expressions are often code.
        if (
            not looks_like_code
            and "(" in text
            and ")" in text
            and word_count <= 12
            and not text.endswith(".")
        ):
            looks_like_code = True

        # Safely inspect surrounding lines. This avoids the
        # "list index out of range" error on the final OCR block.
        previous_text = (
            str(previous_item.get("text", "")).strip()
            if previous_item is not None
            else ""
        )

        next_text = (
            str(next_item.get("text", "")).strip()
            if next_item is not None
            else ""
        )

        # An unclear line located between two code lines is
        # probably part of the same code block.
        if (
            not looks_like_code
            and previous_text
            and next_text
            and looks_like_program_code(previous_text)
            and looks_like_program_code(next_text)
        ):
            looks_like_code = True

        block_type = "paragraph"

        # Page number must be checked before code detection.
        if (
            near_page_bottom
            and (
                numeric_page_number
                or roman_page_number
            )
        ):
            block_type = "page_number"

        elif looks_like_code:
            block_type = "code"

        # Author/byline.
        elif (
            top < page_height * 0.35
            and word_count <= 8
            and len(text) <= 80
            and text.lower().startswith("by ")
        ):
            block_type = "subheading"

        # Main title or section heading.
        elif (
            short_heading_candidate
            and (
                # Main heading near the top of a page.
                (
                    near_page_top
                    and (
                        centered
                        or strongly_larger
                    )
                    and (
                        is_uppercase_title
                        or is_title_case
                        or noticeably_larger
                    )
                )

                # Section heading elsewhere on the page.
                or (
                    (
                        is_uppercase_title
                        or is_title_case
                    )
                    and (
                        noticeably_larger
                        or separated_from_body
                    )
                )
            )
        ):
            block_type = "heading"

        classified_blocks.append({
            **item,
            "text": text,
            "type": block_type,
            "gap_before": gap_before,
            "gap_after": gap_after
        })

    return classified_blocks
