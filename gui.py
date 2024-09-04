import tkinter as tk
import fire
from predictors import predictor_registry

def dumb_autocomplete(text, cursor_pos):
    """
    Autocomplete function that takes the current text and cursor position,
    and returns the autocompleted text.
    For demonstration, it simply appends ' - AutoCompleted' after the word.
    """
    # Find the start of the last word before the cursor position
    words = text[:cursor_pos].split()
    if not words:
        return text, cursor_pos  # No word to complete

    last_word_start = text[:cursor_pos].rfind(words[-1])
    completed_text = text[:cursor_pos] + ' - AutoCompleted' + text[cursor_pos:]
    new_cursor_pos = cursor_pos + len(' - AutoCompleted')
    
    return completed_text, new_cursor_pos

def autocomplete(text, cursor_pos, predictor):
    if not predictor:
        return dumb_autocomplete(text, cursor_pos)
    
    text_before = text[:cursor_pos]
    text_after = text[cursor_pos:]
    completion = predictor.predict(text_after=text_after, text_before=text_before)
    return text_before + completion + text_after, cursor_pos + len(completion)



def main(predictor_name='openai_chat_predictor'):
    if predictor_name != "dumb":
        predictor = predictor_registry[predictor_name]()
    else:
        predictor = None

    # Create the main application window
    root = tk.Tk()
    root.title("Autocomplete Text Box")

    # Create a Text widget
    text_box = tk.Text(root, wrap=tk.WORD)
    text_box.pack(expand=True, fill=tk.BOTH)

    def on_tab(event):
        """
        Event handler for Tab key press to trigger autocompletion.
        """
        # Get the current text and cursor position from the text widget
        current_text = text_box.get("1.0", tk.END)
        cursor_index = text_box.index(tk.INSERT)

        # Convert the cursor position into line and character indices
        row, col = map(int, cursor_index.split('.'))
        cursor_pos = text_box.count("1.0", cursor_index)[0]  # Get absolute cursor index

        # Call the autocomplete function
        autocompleted_text, new_cursor_pos = autocomplete(current_text, cursor_pos, predictor=predictor)

        # Replace the text in the text widget with the autocompleted text
        text_box.delete("1.0", tk.END)
        text_box.insert("1.0", autocompleted_text)

        # Set the new cursor position after autocompletion
        new_row_col = text_box.index(f"1.0 + {new_cursor_pos} chars")
        text_box.mark_set(tk.INSERT, new_row_col)

        # Prevent the default behavior of the Tab key (inserting a tab space)
        return "break"

    # Bind the Tab key to the on_tab function
    text_box.bind("<Tab>", on_tab)

    # Start the Tkinter main event loop
    root.mainloop()

if __name__ == "__main__":
    fire.Fire(main)
