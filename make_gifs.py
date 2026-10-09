import glob
import imageio.v2 as imageio

def make_gif(folder, out_name, fps=3):
    files = sorted(glob.glob(f"{folder}/*.png"))
    images = [imageio.imread(f) for f in files]
    imageio.mimsave(out_name, images, fps=fps, loop=0)
    print(f"saved {out_name} ({len(images)} frames)")

make_gif("out_det", "detection.gif")
make_gif("out_seg", "segmentation.gif")