
from pathlib import Path
import tkinter as tk
from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parent.parent
IMAGE_PATH = ROOT / "data/raw/construction-ppe/train/images/image1008_jpeg.rf.72b87d6b322cd824844053582427c688.jpg"

image = Image.open(IMAGE_PATH)
print("Original image size:", image.size)

scale = 1.0
image.thumbnail((640, 640))

root = tk.Tk()
root.title("Read-only hand inspection")
photo = ImageTk.PhotoImage(image)

label = tk.Label(root, image=photo)
label.pack()

def show_position(event):
    print(f"Displayed position: x={event.x}, y={event.y}")

label.bind("<Button-1>", show_position)

root.mainloop()
