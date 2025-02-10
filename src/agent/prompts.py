# Dummy prompt templates and examples. Replace these with your actual templates.
row_selection_agent_prompt = (
    "Task: {question}\n"
    "{examples}\n"
    "Current scratchpad:\n{scratchpad}\n"
    "Kindly specify the indices of the rows"
)
row_selection_reflect_agent_prompt = (
    "Task: {question}\n"
    "{examples}\n"
    "Previous reflections:\n{reflections}\n"
    "Current scratchpad:\n{scratchpad}\n"
    "What is your next thought and action?"
)
row_selection_reflect_prompt = (
    "Reflect on the previous attempts for the task: {question}\n"
    "Scratchpad:\n{scratchpad}\n"
    "Provide a reflection on how to improve the row selection."
)

ROW_SELECTION_EXAMPLES = (
    "Example: When selecting rows, prefer labeled rows over unlabeled ones to avoid extra cost."
)

ROW_SELECTION_REFLECT_EXAMPLES = (
    "Reflection Example: If too many unlabeled rows were selected (thus incurring high cost), "
    "try to focus on selecting from the labeled rows in the next attempt."
)