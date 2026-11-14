
import json
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from PIL import Image, ImageTk

BASE = Path.cwd()
IMAGE_DIR = BASE / "data/raw/construction-ppe/train/images"
PROPOSAL_PATH = BASE / "data/reports/manual_box_proposals.json"

TARGETS = [
    "image1008_jpeg.rf.72b87d6b322cd824844053582427c688.jpg",
    "image1060_jpeg.rf.94cb69b154d29859341e2bdcb8559ade.jpg",
]
CLASSES = ["no_gloves", "no_goggle", "no_helmet", "no_vest"]


class BoxEditor:
    def __init__(self, root):
        self.root = root
        root.title("Construction PPE — Manual Box Review")
        root.geometry("1200x900")

        self.index = 0
        self.boxes = []
        self.selected = None
        self.mode = "draw"
        self.drag_start = None
        self.drag_action = None
        self.drag_original = None
        self.current_rect = None
        self.scale = 1.0
        self.offset_x = 10
        self.offset_y = 10
        self.photo = None
        self.image = None
        self.path = None
        self.proposals = {}

        if PROPOSAL_PATH.exists():
            saved = json.loads(
                PROPOSAL_PATH.read_text(encoding="utf-8-sig")
            )
            self.proposals = saved.get("images", {})

        bar = ttk.Frame(root, padding=8)
        bar.pack(fill="x")

        self.image_label = ttk.Label(bar, text="")
        self.image_label.pack(side="left", padx=5)

        ttk.Label(bar, text="Class:").pack(side="left", padx=(10, 3))
        self.class_var = tk.StringVar(value="no_gloves")
        ttk.Combobox(
            bar, textvariable=self.class_var,
            values=CLASSES, state="readonly", width=12
        ).pack(side="left")

        ttk.Button(
            bar, text="Draw new box",
            command=self.set_draw_mode
        ).pack(side="left", padx=4)

        ttk.Button(
            bar, text="Select / resize",
            command=self.set_edit_mode
        ).pack(side="left", padx=4)

        ttk.Button(
            bar, text="Delete selected",
            command=self.delete_selected
        ).pack(side="left", padx=4)

        ttk.Button(
            bar, text="Undo new box",
            command=self.undo
        ).pack(side="left", padx=4)

        ttk.Button(
            bar, text="Save proposals",
            command=self.save
        ).pack(side="left", padx=4)

        ttk.Button(
            bar, text="Next image",
            command=self.next_image
        ).pack(side="left", padx=4)

        self.canvas = tk.Canvas(root, bg="#333333", cursor="crosshair")
        self.canvas.pack(fill="both", expand=True, padx=8, pady=8)
        self.canvas.bind("<ButtonPress-1>", self.mouse_down)
        self.canvas.bind("<B1-Motion>", self.mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.mouse_up)

        self.status = ttk.Label(root, padding=8)
        self.status.pack(fill="x")
        self.load_image()

    def load_image(self):
        self.canvas.delete("all")
        self.boxes = []
        self.selected = None
        self.index = min(self.index, len(TARGETS) - 1)
        self.path = IMAGE_DIR / TARGETS[self.index]

        if not self.path.exists():
            messagebox.showerror("Image not found", str(self.path))
            return

        self.image = Image.open(self.path).convert("RGB")
        self.root.update_idletasks()

        max_w = max(300, self.canvas.winfo_width() - 20)
        max_h = max(300, self.canvas.winfo_height() - 20)
        self.scale = min(
            max_w / self.image.width,
            max_h / self.image.height,
            1.5
        )

        display = self.image.resize((
            round(self.image.width * self.scale),
            round(self.image.height * self.scale)
        ))
        self.photo = ImageTk.PhotoImage(display)
        self.offset_x = 10
        self.offset_y = 10
        self.canvas.create_image(
            self.offset_x, self.offset_y,
            image=self.photo, anchor="nw"
        )

        key = f"train/{self.path.name}"
        self.boxes = [
            dict(b) for b in self.proposals.get(key, [])
        ]
        self.image_label.config(
            text=f"{self.index + 1}/{len(TARGETS)}: {self.path.name}"
        )
        self.set_draw_mode()
        self.redraw_boxes()

    def image_point(self, x, y):
        ix = (x - self.offset_x) / self.scale
        iy = (y - self.offset_y) / self.scale
        if 0 <= ix < self.image.width and 0 <= iy < self.image.height:
            return ix, iy
        return None

    def canvas_box(self, box):
        return (
            self.offset_x + box["left"] * self.scale,
            self.offset_y + box["top"] * self.scale,
            self.offset_x + box["right"] * self.scale,
            self.offset_y + box["bottom"] * self.scale
        )

    def set_draw_mode(self):
        self.mode = "draw"
        self.selected = None
        self.redraw_boxes()
        self.status.config(
            text="DRAW mode: choose a class and drag to create a box."
        )

    def set_edit_mode(self):
        self.mode = "edit"
        self.redraw_boxes()
        self.status.config(
            text="EDIT mode: click a box to select it; drag a corner to resize."
        )

    def mouse_down(self, event):
        point = self.image_point(event.x, event.y)
        if point is None:
            return

        if self.mode == "draw":
            self.drag_start = point
            self.current_rect = self.canvas.create_rectangle(
                event.x, event.y, event.x, event.y,
                outline="yellow", width=2
            )
            return

        # Prefer a corner handle; otherwise select a box by clicking inside.
        for i in range(len(self.boxes) - 1, -1, -1):
            x1, y1, x2, y2 = self.canvas_box(self.boxes[i])
            corners = {
                "left": (x1, (y1 + y2) / 2),
                "right": (x2, (y1 + y2) / 2),
                "top": ((x1 + x2) / 2, y1),
                "bottom": ((x1 + x2) / 2, y2),
            }
            for action, (cx, cy) in corners.items():
                if abs(event.x - cx) <= 9 and abs(event.y - cy) <= 9:
                    self.selected = i
                    self.drag_action = action
                    self.drag_start = point
                    self.drag_original = dict(self.boxes[i])
                    self.redraw_boxes()
                    return

        for i in range(len(self.boxes) - 1, -1, -1):
            x1, y1, x2, y2 = self.canvas_box(self.boxes[i])
            if x1 <= event.x <= x2 and y1 <= event.y <= y2:
                self.selected = i
                self.drag_action = None
                self.redraw_boxes()
                return

        self.selected = None
        self.redraw_boxes()

    def mouse_move(self, event):
        if self.drag_start is None:
            return

        if self.mode == "draw" and self.current_rect:
            self.canvas.coords(
                self.current_rect,
                self.offset_x + self.drag_start[0] * self.scale,
                self.offset_y + self.drag_start[1] * self.scale,
                event.x, event.y
            )
            return

        if self.mode != "edit" or self.selected is None:
            return

        point = self.image_point(event.x, event.y)
        if point is None or self.drag_original is None:
            return

        box = dict(self.drag_original)
        ix, iy = point
        if self.drag_action == "left":
            box["left"] = min(round(ix), box["right"] - 3)
        elif self.drag_action == "right":
            box["right"] = max(round(ix), box["left"] + 3)
        elif self.drag_action == "top":
            box["top"] = min(round(iy), box["bottom"] - 3)
        elif self.drag_action == "bottom":
            box["bottom"] = max(round(iy), box["top"] + 3)

        self.boxes[self.selected] = box
        self.redraw_boxes()

    def mouse_up(self, event):
        if self.mode == "draw" and self.drag_start is not None:
            end = self.image_point(event.x, event.y)
            start = self.drag_start
            if end is not None:
                left, right = sorted([round(start[0]), round(end[0])])
                top, bottom = sorted([round(start[1]), round(end[1])])
                if right - left >= 3 and bottom - top >= 3:
                    self.boxes.append({
                        "class": self.class_var.get(),
                        "left": left, "top": top,
                        "right": right, "bottom": bottom
                    })
            if self.current_rect:
                self.canvas.delete(self.current_rect)
            self.current_rect = None
            self.redraw_boxes()

        self.drag_start = None
        self.drag_action = None
        self.drag_original = None

    def redraw_boxes(self):
        self.canvas.delete("review_box")
        for i, box in enumerate(self.boxes):
            x1, y1, x2, y2 = self.canvas_box(box)
            color = "yellow" if i == self.selected else (
                "cyan" if box["class"] == "no_goggle" else "lime"
            )
            self.canvas.create_rectangle(
                x1, y1, x2, y2, outline=color, width=2,
                tags="review_box"
            )
            self.canvas.create_text(
                x1 + 3, y1 + 3, anchor="nw",
                text=f"{i + 1}. {box['class']}",
                fill=color, tags="review_box"
            )

            if i == self.selected and self.mode == "edit":
                for cx, cy in [
                    (x1, (y1 + y2) / 2), (x2, (y1 + y2) / 2),
                    ((x1 + x2) / 2, y1), ((x1 + x2) / 2, y2)
                ]:
                    self.canvas.create_rectangle(
                        cx - 4, cy - 4, cx + 4, cy + 4,
                        fill="white", outline="black", tags="review_box"
                    )

    def delete_selected(self):
        if self.selected is None:
            messagebox.showinfo("Select a box", "Select a box in EDIT mode first.")
            return
        if messagebox.askyesno(
            "Delete proposal",
            "Delete only the selected proposed box? Original labels are untouched."
        ):
            self.boxes.pop(self.selected)
            self.selected = None
            self.redraw_boxes()

    def undo(self):
        if self.boxes:
            self.boxes.pop()
            self.selected = None
            self.redraw_boxes()

    def save(self):
        key = f"train/{self.path.name}"
        self.proposals[key] = [dict(b) for b in self.boxes]
        report = {
            "description": "Proposed boxes only; original YOLO labels are unchanged.",
            "coordinate_format": "pixel xyxy in original image",
            "images": self.proposals,
            "original_labels_modified": False,
            "training_started": False
        }
        PROPOSAL_PATH.parent.mkdir(parents=True, exist_ok=True)
        PROPOSAL_PATH.write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        self.status.config(text=f"Saved proposals: {PROPOSAL_PATH}")
        messagebox.showinfo("Saved", "Proposals saved. Original labels unchanged.")

    def next_image(self):
        if self.index < len(TARGETS) - 1:
            self.index += 1
            self.load_image()
        else:
            messagebox.showinfo("Review complete", "Both images have been shown.")


root = tk.Tk()
app = BoxEditor(root)
root.mainloop()
