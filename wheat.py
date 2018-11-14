from segment_formation import segment_image4


if __name__ == "__main__":
    imgFile = input("Enter the wheat image path : ")
    # for imgFile in imgFile:
    print("L1 Segmentation in process...")
    segment_image4(imgFile)
    print("Segmentation in Complete.")
