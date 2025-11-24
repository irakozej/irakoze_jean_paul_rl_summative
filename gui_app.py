import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import time
import numpy as np
import os
import random

# Set SB3 to use Gymnasium
os.environ['SB3_USE_GYMNASIUM'] = '1'

try:
    from environment.custom_env import AdaptiveLearningEnv
except Exception:
   
    import sys
    sys.path.append(os.getcwd())
    from environment.custom_env import AdaptiveLearningEnv

from stable_baselines3 import DQN, PPO, A2C


class AgentGUI(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.env = None
        self.model = None
        self.algo = None
        self.running = False
        self.paused = False
        self.thread = None
        self.speed = 1.0  
        self._build_ui()
        self.load_default_model()

    def load_default_model(self):
        default_path = "models/ppo/run_12/final_model.zip"
        algo = 'ppo'
        self.txt_log.config(state='normal')
        self.txt_log.insert('end', f'Loading default {algo} model from {default_path}\n')
        self.txt_log.config(state='disabled')
        self.txt_log.see('end')

        self.env = AdaptiveLearningEnv()
        try:
            self.model = PPO.load(default_path)
        except Exception as e:
            self.txt_log.config(state='normal')
            self.txt_log.insert('end', f'Failed to load default model: {e}\n')
            self.txt_log.config(state='disabled')
            self.txt_log.see('end')
            return

        self.algo = algo
        self.reset_env()
        self.txt_log.config(state='normal')
        self.txt_log.insert('end', 'Default model loaded and environment initialized.\n')
        self.txt_log.config(state='disabled')
        self.txt_log.see('end')

    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=0)

        left = ttk.Frame(self)
        left.grid(row=0, column=0, sticky='nsew', padx=8, pady=8)
        left.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(left, width=520, height=420, bg='#f7f7f7', highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky='nsew')

        ctrl = ttk.Frame(left)
        ctrl.grid(row=1, column=0, pady=(8,0), sticky='ew')
        ctrl.columnconfigure(6, weight=1)

        self.btn_load = ttk.Button(ctrl, text='Load Model', command=self.load_model)
        self.btn_load.grid(row=0, column=0, padx=4)

        self.btn_start = ttk.Button(ctrl, text='Start', command=self.start)
        self.btn_start.grid(row=0, column=1, padx=4)

        self.btn_pause = ttk.Button(ctrl, text='Pause', command=self.toggle_pause)
        self.btn_pause.grid(row=0, column=2, padx=4)

        self.btn_reset = ttk.Button(ctrl, text='Reset', command=self.reset_env)
        self.btn_reset.grid(row=0, column=3, padx=4)

        ttk.Label(ctrl, text='Speed:').grid(row=0, column=4, padx=(12,2))
        self.speed_slider = ttk.Scale(ctrl, from_=0.25, to=4.0, orient='horizontal', command=self._on_speed_change)
        self.speed_slider.set(1.0)
        self.speed_slider.grid(row=0, column=5, sticky='ew', padx=4)

        right = ttk.Frame(self)
        right.grid(row=0, column=1, sticky='ns', padx=6, pady=8)
        right.rowconfigure(1, weight=1)

        metrics = ttk.LabelFrame(right, text='Metrics', padding=(8,8))
        metrics.grid(row=0, column=0, sticky='n', pady=(0,8))

        self.lbl_episode = ttk.Label(metrics, text='Episode: 0')
        self.lbl_episode.grid(row=0, column=0, sticky='w')
        self.lbl_step = ttk.Label(metrics, text='Step: 0')
        self.lbl_step.grid(row=1, column=0, sticky='w')
        self.lbl_reward = ttk.Label(metrics, text='Reward: 0.00')
        self.lbl_reward.grid(row=2, column=0, sticky='w')
        self.lbl_outcome = ttk.Label(metrics, text='Outcome: -')
        self.lbl_outcome.grid(row=3, column=0, sticky='w')

        log_frame = ttk.LabelFrame(right, text='Terminal Log', padding=(8,8))
        log_frame.grid(row=1, column=0, sticky='nsew')
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)

        self.txt_log = tk.Text(log_frame, width=40, height=22, wrap='none')
        self.txt_log.grid(row=0, column=0, sticky='nsew')
        self.txt_log.insert('end', 'Ready. Load a model to begin.\n')
        self.txt_log.config(state='disabled')

        self._draw_static()

    def _draw_static(self):
        self.canvas.delete('all')
        self.canvas.create_rectangle(10, 10, 510, 70, fill='#e9eef8', outline='')
        self.canvas.create_text(20, 20, anchor='nw', text='Adaptive Learning Resource Navigator', font=('Helvetica', 18, 'bold'))

        self.resource_rects = []
        y = 90
        for i, label in enumerate(['Beginner', 'Intermediate', 'Advanced', 'Practice', 'Remediation', 'Break', 'Assess']):
            rect = self.canvas.create_rectangle(20, y, 500, y+50, fill='#ffffff', outline='#cccccc')
            txt = self.canvas.create_text(260, y+25, anchor='center', text=f'{i}: {label}', font=('Helvetica', 16, 'bold'), fill='#000080')
            self.resource_rects.append((rect, txt))
            y += 60

        self.canvas.create_rectangle(20, 460, 500, 490, fill='#e6e6e6', outline='')
        self.canvas.create_text(20, 470, anchor='w', text='Mastery Progress', font=('Helvetica', 10))
        self.progress_bar = self.canvas.create_rectangle(25, 465, 25, 485, fill='#4caf50', outline='')

    def load_model(self):
        path = filedialog.askopenfilename(title='Select model file (final_model.zip)', filetypes=[('Zip', '*.zip'), ('All files', '*.*')])
        if not path:
            return

        if 'dqn' in path.lower():
            algo = 'dqn'
        elif 'ppo' in path.lower():
            algo = 'ppo'
        elif 'a2c' in path.lower():
            algo = 'a2c'
        else:
           
            algo = tk.simpledialog.askstring('Algorithm', 'Enter algorithm (dqn/ppo/a2c)')
            if not algo:
                return
        self.txt_log.config(state='normal')
        self.txt_log.insert('end', f'Loading {algo} model from {path}\n')
        self.txt_log.config(state='disabled')
        self.txt_log.see('end')

        # load env
        self.env = AdaptiveLearningEnv()
        try:
            if algo == 'dqn':
                self.model = DQN.load(path)
            elif algo == 'ppo':
                self.model = PPO.load(path)
            elif algo == 'a2c':
                self.model = A2C.load(path)
            else:
                raise ValueError('Unsupported algorithm')
        except Exception as e:
            self.txt_log.config(state='normal')
            self.txt_log.insert('end', f'Failed to load model: {e}\n')
            self.txt_log.config(state='disabled')
            self.txt_log.see('end')
            return

        self.algo = algo
        self.reset_env()
        self.txt_log.config(state='normal')
        self.txt_log.insert('end', 'Model loaded and environment initialized.\n')
        self.txt_log.config(state='disabled')
        self.txt_log.see('end')

    def start(self):
        if self.model is None or self.env is None:
            messagebox.showwarning('No model', 'Load a trained model first')
            return
        if self.running:
            return
        self.running = True
        self.paused = False
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        self.txt_log.config(state='normal')
        self.txt_log.insert('end', 'Simulation started.\n')
        self.txt_log.config(state='disabled')
        self.txt_log.see('end')

    def toggle_pause(self):
        if not self.running:
            return
        self.paused = not self.paused
        self.txt_log.config(state='normal')
        self.txt_log.insert('end', f'Paused: {self.paused}\n')
        self.txt_log.config(state='disabled')
        self.txt_log.see('end')

    def reset_env(self):
        # reset environment and UI
        if self.env is None:
            self.env = AdaptiveLearningEnv()
        obs = self.env.reset()
        if isinstance(obs, tuple) or isinstance(obs, list):
           
            obs = obs[0]
        self.current_obs = obs
        self.episode = 0
        self.step = 0
        self.episode_reward = 0.0
        self.update_metrics(self.episode, self.step, self.episode_reward, '-')
        self.highlight_action(None)
        self.txt_log.config(state='normal')
        self.txt_log.insert('end', 'Environment reset.\n')
        self.txt_log.config(state='disabled')
        self.txt_log.see('end')

    def _on_speed_change(self, val):
        try:
            self.speed = float(val)
        except Exception:
            self.speed = 1.0

    def _run_loop(self):
        while self.running:
            if self.paused:
                time.sleep(0.1)
                continue

            action = random.randint(0, 6)
            result = self.env.step(int(action))
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
                done = bool(terminated or truncated)
            elif len(result) == 4:
                obs, reward, done, info = result
            else:
                obs = result[0]
                reward = result[1]
                done = bool(result[2])
                info = result[3] if len(result) > 3 else {}

            self.current_obs = obs
            self.episode_reward += float(reward)
            self.step += 1

            self.master.after(0, self.update_ui, action, reward, done, info)

            delay = max(0.01, 0.25 / self.speed)
            time.sleep(delay)

            if done:
                outcome = info.get('outcome', 'done') if isinstance(info, dict) else 'done'
                self.txt_log.config(state='normal')
                self.txt_log.insert('end', f'Episode {self.episode} finished. Reward={self.episode_reward:.3f}, Outcome={outcome}\n')
                self.txt_log.config(state='disabled')
                self.txt_log.see('end')
                self.episode += 1
                self.env.reset()
                self.current_obs = self.env.reset()
                if isinstance(self.current_obs, tuple) or isinstance(self.current_obs, list):
                    self.current_obs = self.current_obs[0]
                self.episode_reward = 0.0
                self.step = 0

    def update_ui(self, action, reward, done, info):
        # update metrics
        self.update_metrics(self.episode, self.step, self.episode_reward, info.get('outcome') if isinstance(info, dict) else '-')

        try:
            self.highlight_action(int(action))
        except Exception:
            self.highlight_action(None)

    def update_metrics(self, episode, step, reward, outcome):
        self.lbl_episode.config(text=f'Episode: {episode}')
        self.lbl_step.config(text=f'Step: {step}')
        self.lbl_reward.config(text=f'Reward: {reward:.3f}')
        self.lbl_outcome.config(text=f'Outcome: {outcome}')

    def highlight_action(self, idx):
        # reset all
        for i, (rect, txt) in enumerate(self.resource_rects):
            self.canvas.itemconfig(rect, fill='#ffffff')
        if idx is None:
            return
        if idx >= 0 and idx < len(self.resource_rects):
            rect, txt = self.resource_rects[idx]
            self.canvas.itemconfig(rect, fill='#d5f5d5')


# simple run helper
if __name__ == '__main__':
    root = tk.Tk()
    root.title('RL GUI Test')
    app = AgentGUI(master=root)
    app.pack(fill='both', expand=True)
    root.mainloop()
