from pathlib import Path
import cv2
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET = PROJECT_ROOT / "data" / "processed" / "construction-vehicle-detection-clean"
OUTPUT = Path(__file__).resolve().parents[1] / "research" / "roller_loader_box_proposals.json"

TARGETS = [
    ("train", "Heavy_Equipment_5331_jpg.rf.2ae911120fdb0ebbd7cf561e28df90eb"),
    ("train", "Heavy_Equipment_5914_jpg.rf.462a48e2d23dd78b8fed0d595efef471"),
    ("train", "Heavy_Equipment_5933_jpg.rf.3a2f4b894300e3448d0894f1fe02e71f"),
    ("train", "Heavy_Equipment_5933_jpg.rf.7406b5491e4f5b2362b477512cd9829c"),
    ("valid", "Heavy_Equipment_6198_jpg.rf.c0bf682e109f080a13187b38a9d72af1"),
]

def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Output already exists; refusing to overwrite: {OUTPUT}")

    proposals = []

    for split, stem in TARGETS:
        image_path = DATASET / split / "images" / f"{stem}.jpg"
        label_path = DATASET / split / "labels" / f"{stem}.txt"

        if not image_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(f"Missing image or label for {stem}")

        original = label_path.read_text(encoding="utf-8").strip()
        if original != "9 0.5 0.5 1 1":
            raise ValueError(f"Unexpected original label for {stem}: {original!r}")

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        height, width = image.shape[:2]
        if (width, height) != (640, 480):
            raise ValueError(f"Unexpected dimensions for {stem}: {width}x{height}")

        while True:
            print(f"\nImage: {stem}")
            print("Draw one tight rectangle around the Loader.")
            print("Press ENTER or SPACE to accept the rectangle; ESC to cancel.")

            x, y, w, h = cv2.selectROI(
                "Draw Loader bounding box",
                image,
                showCrosshair=True,
                fromCenter=False
            )

            if w <= 0 or h <= 0:
                cv2.destroyAllWindows()
                print("Annotation cancelled. No output file was written.")
                return

            preview = image.copy()
            cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                preview, "Loader - accept: Y | redraw: R | quit: Q",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2
            )
            cv2.imshow("Review Loader box", preview)
            key = cv2.waitKey(0) & 0xFF
            cv2.destroyWindow("Review Loader box")

            if key == ord("q"):
                cv2.destroyAllWindows()
                print("Stopped. No output file was written.")
                return
            if key == ord("r"):
                continue
            if key == ord("y"):
                break

        x_center = (x + w / 2) / width
        y_center = (y + h / 2) / height
        box_width = w / width
        box_height = h / height

        proposals.append({
            "split": split,
            "image": image_path.relative_to(PROJECT_ROOT).as_posix(),
            "label": label_path.relative_to(PROJECT_ROOT).as_posix(),
            "original_label": original,
            "proposed_label": (
                f"6 {x_center:.8f} {y_center:.8f} "
                f"{box_width:.8f} {box_height:.8f}"
            ),
            "image_width": width,
            "image_height": height,
            "reviewed": True
        })
        print(f"Accepted proposal for {stem}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(proposals, indent=2), encoding="utf-8")
    cv2.destroyAllWindows()

    print(f"\nCompleted: {len(proposals)} Loader boxes")
    print(f"Proposals saved to: {OUTPUT}")
    print("Original dataset images and labels were not modified.")

if __name__ == "__main__":
    main()
