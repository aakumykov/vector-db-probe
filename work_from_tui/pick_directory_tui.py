import os
from simple_term_menu import TerminalMenu

def pick_directory_tui(start_dir="."):
    current_dir = os.path.abspath(start_dir)
    
    while True:
        try:
            # Get all subdirectories, ignoring hidden ones
            entries = os.listdir(current_dir)
            dirs = sorted([d for d in entries if os.path.isdir(os.path.join(current_dir, d)) and not d.startswith('.')])
        except PermissionError:
            dirs = []
        
        # Create menu options
        menu_entries = ["[..]"] + [f"[{d}]" for d in dirs] + ["[✓ Выбрать текущий ("+current_dir+")]"]
        
        terminal_menu = TerminalMenu(
            menu_entries,
            title=f"Текущий каталог: {current_dir}",
            clear_screen=False,
            show_search_hint=True,
            cycle_cursor=True
        )
        
        choice = terminal_menu.show()
        
        # Handle selection
        if choice == 0:
            # Go up (stop if we are at the root directory)
            parent = os.path.dirname(current_dir)
            if parent != current_dir:
                current_dir = parent
        elif choice == len(menu_entries) - 1:
            # User selected the current directory
            return current_dir
        else:
            # Go into the selected subdirectory
            current_dir = os.path.join(current_dir, dirs[choice - 1])


# --- Usage in your script ---
if __name__ == "__main__":
    print("Starting directory picker...")
    selected_folder = pick_directory_tui(start_dir=os.path.expanduser("~")) # Starts in Home folder
    print(f"\nYou selected: {selected_folder}")
