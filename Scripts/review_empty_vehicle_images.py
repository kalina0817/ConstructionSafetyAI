
from __future__ import annotations

import json
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from datetime import datetime

try:
    import yaml
    from PIL import Image, ImageTk
except ImportError as exc:
    raise SystemExit(
        "Install the required packages with:\n"
        "python -m pip install pyyaml pillow"
    ) from exc


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = (
    PROJECT_ROOT / "data" / "processed" / "vehicle_detector"
)

IMAGE_DIR = DATASET_ROOT / "train" / "images"
LABEL_DIR = DATASET_ROOT / "train" / "labels"
DATA_YAML = DATASET_ROOT / "data.yaml"

REVIEW_DIR = PROJECT_ROOT / "data" / "review"
REVIEW_FILE = REVIEW_DIR / "vehicle_empty_label_manual_review.json"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

def load_class_names():
    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Dataset configuration not found:\n{DATA_YAML}"
        )

    with DATA_YAML.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    names = config.get("names", [])

    if isinstance(names, dict):
        names = [
            names[key]
            for key in sorted(names, key=lambda item: int(item))
        ]

    if not names:
        raise ValueError("No class names found in data.yaml.")

    return [str(name) for name in names]


def find_image(label_stem):
    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Image directory not found:\n{IMAGE_DIR}"
        )

    for image_path in IMAGE_DIR.iterdir():
        if (
            image_path.is_file()
            and image_path.stem == label_stem
            and image_path.suffix.lower() in IMAGE_EXTENSIONS
        ):
            return image_path

    return None


def find_empty_label_images():
    if not LABEL_DIR.exists():
        raise FileNotFoundError(
            f"Label directory not found:\n{LABEL_DIR}"
        )

    items = []

    for label_path in sorted(LABEL_DIR.glob("*.txt")):
        # Only include genuinely empty label files.
        content = label_path.read_text(
            encoding="utf-8",
            errors="ignore"
        ).strip()

        if content:
            continue

        image_path = find_image(label_path.stem)

        if image_path is not None:
            items.append({
                "label_path": label_path,
                "image_path": image_path,
            })

    if not items:
        raise RuntimeError(
            "No empty label files with matching images were found."
        )

    return items


def load_existing_reviews():
    if not REVIEW_FILE.exists():
        return {}

    try:
        with REVIEW_FILE.open("r", encoding="utf-8") as file:
            data = json.load(file)

        return data if isinstance(data, dict) else {}

    except (OSError, json.JSONDecodeError):
        messagebox.showwarning(
            "Review file warning",
            "The existing review file could not be read. "
            "A new review file will be created when you save."
        )
        return {}


# --------------------------------------------------
# APPLICATION
# --------------------------------------------------

def main():
    classes = load_class_names()
    items = find_empty_label_images()

    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    reviews = load_existing_reviews()

    root = tk.Tk()
    root.title("Construction Vehicle — Empty Label Review")
    root.geometry("1250x900")
    root.minsize(900, 650)

    # Shared application state.
    current_index = 0
    photo = None
    class_vars = {}

    status_var = tk.StringVar(value="uncertain")

    # Main window layout.
    root.columnconfigure(0, weight=1)
    root.rowconfigure(1, weight=1)

    header = ttk.Frame(root, padding=10)
    header.grid(row=0, column=0, sticky="ew")
    header.columnconfigure(0, weight=1)

    ttk.Label(
        header,
        text="Review Empty Vehicle Labels",
        font=("Segoe UI", 15, "bold")
    ).grid(row=0, column=0, sticky="w")

    progress_label = ttk.Label(header, text="")
    progress_label.grid(row=0, column=1, sticky="e")

    body = ttk.Frame(root, padding=(10, 0, 10, 8))
    body.grid(row=1, column=0, sticky="nsew")

    body.columnconfigure(0, weight=3)
    body.columnconfigure(1, weight=2)
    body.rowconfigure(0, weight=1)

    # --------------------------------------------------
    # IMAGE PANEL
    # --------------------------------------------------

    image_panel = ttk.LabelFrame(
        body,
        text="Image being reviewed",
        padding=8
    )
    image_panel.grid(
        row=0,
        column=0,
        sticky="nsew",
        padx=(0, 10)
    )

    image_panel.columnconfigure(0, weight=1)
    image_panel.rowconfigure(1, weight=1)

    image_name_label = ttk.Label(
        image_panel,
        text="",
        wraplength=700
    )
    image_name_label.grid(
        row=0,
        column=0,
        sticky="ew",
        pady=(0, 8)
    )

    image_display = ttk.Label(
        image_panel,
        anchor="center"
    )
    image_display.grid(
        row=1,
        column=0,
        sticky="nsew"
    )

    # --------------------------------------------------
    # REVIEW PANEL
    # --------------------------------------------------

    side_panel = ttk.Frame(body)
    side_panel.grid(row=0, column=1, sticky="nsew")

    side_panel.columnconfigure(0, weight=1)
    side_panel.rowconfigure(1, weight=1)

    # Image-level decision.
    status_frame = ttk.LabelFrame(
        side_panel,
        text="What is visible?",
        padding=8
    )
    status_frame.grid(
        row=0,
        column=0,
        sticky="ew",
        pady=(0, 8)
    )

    ttk.Radiobutton(
        status_frame,
        text="Construction vehicle is visible",
        variable=status_var,
        value="vehicle_present"
    ).pack(anchor="w", pady=3)

    ttk.Radiobutton(
        status_frame,
        text="No construction vehicle is visible",
        variable=status_var,
        value="no_vehicle"
    ).pack(anchor="w", pady=3)

    ttk.Radiobutton(
        status_frame,
        text="Uncertain / cannot tell",
        variable=status_var,
        value="uncertain"
    ).pack(anchor="w", pady=3)

    # Scrollable list of all dataset classes.
    classes_frame = ttk.LabelFrame(
        side_panel,
        text="Select visible vehicle classes",
        padding=8
    )
    classes_frame.grid(
        row=1,
        column=0,
        sticky="nsew",
        pady=(0, 8)
    )

    classes_frame.columnconfigure(0, weight=1)
    classes_frame.rowconfigure(0, weight=1)

    class_canvas = tk.Canvas(
        classes_frame,
        highlightthickness=0
    )

    class_scrollbar = ttk.Scrollbar(
        classes_frame,
        orient="vertical",
        command=class_canvas.yview
    )

    checkbox_frame = ttk.Frame(class_canvas)
    checkbox_frame.columnconfigure(0, weight=1)

    canvas_window = class_canvas.create_window(
        (0, 0),
        window=checkbox_frame,
        anchor="nw"
    )

    class_canvas.configure(
        yscrollcommand=class_scrollbar.set
    )

    class_canvas.grid(
        row=0,
        column=0,
        sticky="nsew"
    )

    class_scrollbar.grid(
        row=0,
        column=1,
        sticky="ns"
    )

    def update_scroll_region(_event=None):
        class_canvas.configure(
            scrollregion=class_canvas.bbox("all")
        )

    def resize_checkbox_frame(event):
        class_canvas.itemconfigure(
            canvas_window,
            width=event.width
        )

    checkbox_frame.bind(
        "<Configure>",
        update_scroll_region
    )

    class_canvas.bind(
        "<Configure>",
        resize_checkbox_frame
    )

    # One class per row keeps names readable.
    for index, class_name in enumerate(classes):
        variable = tk.BooleanVar(value=False)
        class_vars[class_name] = variable

        ttk.Checkbutton(
            checkbox_frame,
            text=class_name,
            variable=variable
        ).grid(
            row=index,
            column=0,
            sticky="ew",
            padx=5,
            pady=4
        )

    # Mouse wheel scrolling for the class list.
    def scroll_classes(event):
        if getattr(event, "delta", 0):
            class_canvas.yview_scroll(
                int(-event.delta / 120),
                "units"
            )

    class_canvas.bind(
        "<MouseWheel>",
        scroll_classes
    )

    # Notes field.
    notes_frame = ttk.LabelFrame(
        side_panel,
        text="Notes (optional)",
        padding=8
    )
    notes_frame.grid(
        row=2,
        column=0,
        sticky="ew"
    )

    notes_frame.columnconfigure(0, weight=1)

    notes_text = tk.Text(
        notes_frame,
        height=4,
        wrap="word"
    )
    notes_text.grid(
        row=0,
        column=0,
        sticky="ew"
    )

    # --------------------------------------------------
    # SAVE AND DISPLAY FUNCTIONS
    # --------------------------------------------------

    def review_key(item):
        return item["image_path"].name

    def save_current(show_message=False):
        item = items[current_index]

        selected_classes = [
            name
            for name, variable in class_vars.items()
            if variable.get()
        ]

        reviews[review_key(item)] = {
            "image": item["image_path"].relative_to(
                PROJECT_ROOT
            ).as_posix(),

            "original_empty_label": item["label_path"].relative_to(
                PROJECT_ROOT
            ).as_posix(),

            "status": status_var.get(),
            "visible_classes": selected_classes,
            "notes": notes_text.get(
                "1.0",
                "end-1c"
            ).strip(),

            "reviewed_at": datetime.now().astimezone().isoformat(
                timespec="seconds"
            ),
        }

        # Only the separate review JSON is written.
        # Original images and YOLO labels are never edited.
        with REVIEW_FILE.open("w", encoding="utf-8") as file:
            json.dump(
                reviews,
                file,
                indent=2,
                ensure_ascii=False
            )

        if show_message:
            messagebox.showinfo(
                "Saved",
                f"Review saved successfully:\n{REVIEW_FILE}"
            )

    def show_current():
        # Correctly shares the photo variable from main().
        nonlocal photo

        item = items[current_index]
        image_path = item["image_path"]

        progress_label.config(
            text=f"Image {current_index + 1} of {len(items)}"
        )

        image_name_label.config(
            text=image_path.name
        )

        root.update_idletasks()

        max_width = max(
            image_panel.winfo_width() - 40,
            350
        )

        max_height = max(
            image_panel.winfo_height() - 100,
            300
        )

        try:
            with Image.open(image_path) as source:
                image = source.convert("RGB")

                image.thumbnail(
                    (max_width, max_height),
                    Image.Resampling.LANCZOS
                )

                photo = ImageTk.PhotoImage(image.copy())

            image_display.configure(
                image=photo,
                text=""
            )

        except Exception as exc:
            photo = None

            image_display.configure(
                image="",
                text=f"Could not display image:\n{exc}"
            )

        # Restore any previous decision for this image.
        saved = reviews.get(
            review_key(item),
            {}
        )

        status_var.set(
            saved.get("status", "uncertain")
        )

        selected_classes = set(
            saved.get("visible_classes", [])
        )

        for name, variable in class_vars.items():
            variable.set(name in selected_classes)

        notes_text.delete("1.0", "end")
        notes_text.insert(
            "1.0",
            saved.get("notes", "")
        )

    def previous_image():
        nonlocal current_index

        save_current()

        if current_index > 0:
            current_index -= 1
            show_current()

    def next_image():
        nonlocal current_index

        save_current()

        if current_index < len(items) - 1:
            current_index += 1
            show_current()
        else:
            messagebox.showinfo(
                "Review complete",
                "You have reached the last image.\n"
                "Your decisions have been saved."
            )

    def close_app():
        try:
            save_current()
        except OSError as exc:
            if not messagebox.askyesno(
                "Save error",
                f"Could not save the review:\n{exc}\n\nClose anyway?"
            ):
                return

        root.destroy()

    # --------------------------------------------------
    # NAVIGATION BUTTONS
    # --------------------------------------------------

    footer = ttk.Frame(root, padding=10)
    footer.grid(row=2, column=0, sticky="ew")
    footer.columnconfigure(0, weight=1)

    ttk.Label(
        footer,
        text=(
            "Review decisions are saved separately. "
            "Original images and label files remain unchanged."
        ),
        wraplength=500
    ).grid(
        row=0,
        column=0,
        sticky="w"
    )

    buttons = ttk.Frame(footer)
    buttons.grid(
        row=0,
        column=1,
        sticky="e"
    )

    ttk.Button(
        buttons,
        text="Previous",
        command=previous_image
    ).pack(side="left", padx=3)

    ttk.Button(
        buttons,
        text="Save",
        command=lambda: save_current(True)
    ).pack(side="left", padx=3)

    ttk.Button(
        buttons,
        text="Save and Next",
        command=next_image
    ).pack(side="left", padx=3)

    ttk.Button(
        buttons,
        text="Save and Close",
        command=close_app
    ).pack(side="left", padx=3)

    root.protocol(
        "WM_DELETE_WINDOW",
        close_app
    )

    show_current()
    root.mainloop()


if __name__ == "__main__":
    main()
