from handle_ch01 import adjust_ch01_image


def main():
    adjust_ch01_image(
        save_path="adjusted_ch01.png",
        ch01_path="Project001_qCAPs_ACPEGN3_4conc_0,0.1,1,4mgmL_A1 (2)_Merged_ch01.tif",
        exposure=3.0,
        contrast_gain=1.0,
        stretch_low=0.175,
        stretch_high=99.825,
        do_stretch=True,
    )


if __name__ == "__main__":
    main()
