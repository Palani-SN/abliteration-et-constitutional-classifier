import textwrap

# =============================================================================
# CONSOLE RECORD PRINTING
# One bordered block per prompt instead of a run-on wall of text: a header
# line (index/category/verdict), then aligned "label : value" fields, with
# long or multi-line values wrapped and indented under their label instead of
# relying on the terminal's raw line wrap (which doesn't indent continuations,
# so a long response and the next field bleed into each other visually).
# =============================================================================

WIDTH = 100


def print_record(title, fields):
    """
    title:  header string for this record, e.g. "[3/10] HARMFUL  verdict=BLOCK"
    fields: list of (label, value) tuples, printed in order.
    """
    print("\n" + "─" * WIDTH)
    print(title)
    print("─" * WIDTH)

    label_width = max(len(label) for label, _ in fields) + 1
    for label, value in fields:
        text = "(none)" if value is None else str(value)
        fits_inline = "\n" not in text and len(text) <= (WIDTH - label_width - 2)
        if fits_inline:
            print(f"{label:<{label_width}}: {text}")
        else:
            print(f"{label:<{label_width}}:")
            wrapped = textwrap.fill(
                text, width=WIDTH - 4,
                initial_indent="    ", subsequent_indent="    ",
            )
            print(wrapped if wrapped else "    (empty)")
