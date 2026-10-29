"""Reusable Tk sliders for MuJoCo velocity and AMP deployment.

Call ``read()`` from the simulation's main thread at the control rate. It pumps
GUI events and returns ``(vx, vy, wz)``; no GUI thread or mainloop is needed.
"""

import tkinter as tk


class VelocityCommandPanel:
    def __init__(self, command=(0.0, 0.0, 0.0), ranges=None):
        ranges = ranges or {"vx": (-0.5, 2.0), "vy": (-0.3, 0.3), "wz": (0.0, 0.2)}
        self.root = tk.Tk()
        self.root.title("Velocity command")
        self.root.resizable(False, False)
        self.running = True
        self.sliders = []
        for row, (axis, unit, value) in enumerate(zip(("vx", "vy", "wz"), ("m/s", "m/s", "rad/s"), command)):
            tk.Label(self.root, text=f"{axis} ({unit})", width=12).grid(row=row, column=0, padx=8)
            slider = tk.Scale(
                self.root, from_=ranges[axis][0], to=ranges[axis][1],
                resolution=0.01, orient=tk.HORIZONTAL, length=280,
            )
            slider.set(value)
            slider.grid(row=row, column=1, padx=8, pady=4)
            self.sliders.append(slider)
        tk.Button(self.root, text="Zero command (Space)", command=self.zero).grid(
            row=3, column=0, columnspan=2, padx=8, pady=10, sticky="ew"
        )
        self.root.bind("<space>", self.zero)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def read(self):
        if not self.running:
            return (0.0, 0.0, 0.0)
        self.root.update_idletasks()
        self.root.update()
        return tuple(slider.get() for slider in self.sliders) if self.running else (0.0, 0.0, 0.0)

    def zero(self, event=None):
        for slider in self.sliders:
            slider.set(0.0)
        return "break"

    def close(self):
        if self.running:
            self.running = False
            self.root.destroy()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
