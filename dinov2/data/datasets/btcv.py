


def make_splits(data_dir = "/mnt/z/data/Abdomen/RawData")
    train_path = data_dir + os.sep + "train"
    test_path = data_dir + os.sep + "test"
    if "Training" in os.listdir(data_dir): 
        os.rename(data_dir + os.sep + "Training", train_path)
    if "Testing" in os.listdir(data_dir): 
        os.rename(data_dir + os.sep + "Testing", test_path)


    train_image_path = train_path + os.sep + "img"
    train_label_path = train_path + os.sep + "label"
    train_images, train_labels = os.listdir(train_image_path), os.listdir(train_label_path)

    val_path = data_dir + os.sep + "val"
    val_image_path = val_path + os.sep + "img"
    val_label_path = val_path + os.sep + "label"
    os.makedirs(val_path, exist_ok=True), os.makedirs(val_image_path, exist_ok=True), os.makedirs(val_label_path, exist_ok=True)

    val_image_set = [img for i, img in enumerate(train_images) if i % 2 == 0]
    for img in train_images:
        if img in val_image_set:
            shutil.move(train_image_path + os.sep + img, val_image_path)

    val_label_set = [img for i, img in enumerate(train_labels) if i % 2 == 0]
    for label in train_labels:
        if label in val_label_set:
            shutil.move(train_label_path + os.sep + label, val_label_path)