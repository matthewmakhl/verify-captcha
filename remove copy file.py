from pathlib import Path

data_dir = Path("./Images/")
images = sorted(list(map(str, list(data_dir.glob("*.jfif")))))


for img_path in images:
  if "Copy" in img_path:
    file_to_rem = Path(img_path)
    file_to_rem.unlink()