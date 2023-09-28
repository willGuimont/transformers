import torch


class ToyDetectionDataset:
    def __init__(self, image_size: int, num_objects: int, num_images: int):
        """
        Toy detection dataset.
        The dataset consists of binary images with rectangles, and detection targets are the coordinates of the
        rectangles.
        :param image_size: size of the image
        :param num_objects: number of objects in the image
        :param num_images: size of the dataset
        """
        self.image_size = image_size
        self.num_objects = num_objects
        self.cache = {}
        self.num_images = num_images

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        """
        Generate an image and its detection target.
        :param idx: index of the image
        :return: image and its detection target
        """
        if idx in self.cache:
            return self.cache[idx]

        image = torch.zeros((1, self.image_size, self.image_size))
        boxes = torch.zeros((self.num_objects, 4))
        for i in range(self.num_objects):
            x1, y1 = torch.randint(0, self.image_size - 1, (2,))
            x2, y2 = torch.randint(x1 + 1, self.image_size, (2,))
            image[:, y1:y2, x1:x2] = 1

            cx, cy = (x1 + x2) / 2 / self.image_size, (y1 + y2) / 2 / self.image_size
            w, h = (x2 - x1) / self.image_size, (y2 - y1) / self.image_size
            boxes[i] = torch.tensor([cx, cy, w, h])

        self.cache[idx] = (image, boxes)
        return image, boxes
