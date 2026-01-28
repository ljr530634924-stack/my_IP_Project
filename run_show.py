from show_ch01 import adjust_ch01_image


def main():
    adjust_ch01_image(
        r"D:\Ingenieurpraixs\test\Project003_CRP_Fullwell_TileScan 3_CRP_Alexa555_directBinding_B_4Biotin_0-10mg_1_Merged_ch01.tif",
        r"D:\Ingenieurpraixs\test\Test_show_ch01.png",
        exposure=8.0,
        contrast_gain=1.2,
        stretch_mode="percentile",
        std_lo_sigma=1.0,
        std_hi_sigma=10.0,
        use_median_blur=True,
        use_bg_subtract=True,
        bg_sigma=35.0,
        use_petri_mask=True,
    )


if __name__ == "__main__":
    main()
