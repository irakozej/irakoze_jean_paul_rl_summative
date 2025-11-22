# main.py
# Improved GUI with simulation speed control and better styling

import tkinter as tk
from tkinter import ttk
from gui_app import AgentGUI


def main():
    root = tk.Tk()
    root.title("Reinforcement Learning Agent Simulator")
    root.geometry("900x650")

    app = AgentGUI(root)
    app.pack(fill="both", expand=True)

    root.mainloop()


if __name__ == "__main__":
    main()
