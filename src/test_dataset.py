from dataset import SneakerDataset

dataset = SneakerDataset(
    csv_file="data/metadata.csv",
    image_dir="data/filtered_images"
)
print("Dataset size: ", len(dataset))
img,label = dataset[0]
print("Image shape: ", img.shape)
print("Label:",label)