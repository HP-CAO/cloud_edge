"""
To have a proper display with nice fonts, please use system python
"""

import threading
import time
import tkinter as tk
import redis
import pickle

evaluation = False
training_steps = 0
running_mode = "Calibrating"
channel_name = "ch_training_steps"

redis_connection_pool = redis.ConnectionPool(host="10.162.12.241", port="6379", password="cps123456")
redis_connection = redis.Redis(connection_pool=redis_connection_pool)


def subscribe():
    substates = redis_connection.pubsub()
    substates.subscribe(channel_name)
    substates.parse_response()
    return substates


def receive_data():
    step_subscriber = subscribe()
    global training_steps
    global running_mode

    while True:
        pack = step_subscriber.parse_response()[2]
        mode, steps = pickle.loads(pack)
        if mode == "Resetting":
            running_mode = mode
        else:
            if steps <= 3500:
                running_mode = "Collecting Experiences"
            elif steps <= 5000:
                running_mode = "Training Critic"
            else:
                running_mode = mode

        training_steps = steps
        time.sleep(0.001)


def update_display():
    s = "Cumulative Steps:"
    s_1 = "{:,}".format(training_steps)
    s_2 = "{}".format(running_mode)
    l.config(text=s, fg='black', bg='white')
    l1.config(text=s_1, fg='black', bg='white')
    l2.config(text=s_2, fg='black', bg='white')
    root.after(1, update_display)


root = tk.Tk()
root.wm_overrideredirect(True)
root.geometry("{0}x{1}+0+0".format(int(root.winfo_screenwidth() / 2), root.winfo_screenheight()))
root.configure(bg='white')
root.bind("<Button-1>", lambda evt: root.destroy())

l = tk.Label(text='', font=("Arial", 120, "bold"))
l1 = tk.Label(text='', font=("Arial", 160, "bold"))
l2 = tk.Label(text='', font=("Arial", 120, "bold"))

l.pack(expand=True)
l1.pack(expand=True)
l2.pack(expand=True)

thread = threading.Thread(target=receive_data)
thread.start()
update_display()
root.mainloop()


