import cv2

files = [
    "Project001_Image 2_100ugAb_EthanolAmine_ch01.tif",  # 按需要改成你的路径
    "B.png",
]

for p in files:
    img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(p, "-> cannot read")
        continue
    print(p, "dtype:", img.dtype, "shape:", img.shape,
          "min:", int(img.min()), "max:", int(img.max()))
