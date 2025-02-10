def rows_to_markdown(table):
    """
    Converts a 2D list (table) to Markdown table format.

    :param table: List of lists where each sublist represents a row in the table.
    :return: String representing the table in Markdown format.
    """
    if not table:
        return ""

    col_widths = [max(len(str(cell)) for cell in col) for col in zip(*table)]
    header = "| " + " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(table[0])) + " |"
    separator = "|-" + "-|-".join("-" * width for width in col_widths) + "-|"
    rows = [
        "| " + " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row)) + " |"
        for row in table[1:]
    ]
    markdown_table = "\n".join([header, separator] + rows)
    
    return markdown_table