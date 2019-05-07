import os
from glob import glob
import re
import torch.nn as nn

class Utils():

    def normalize(img, mean, std):
        img = img/255.0
        img[0] = (img[0] - mean[0]) / std[0]
        img[1] = (img[1] - mean[1]) / std[1]
        img[2] = (img[2] - mean[2]) / std[2]
        img = np.clip(img, 0.0, 1.0)

        return img

    def denormalize(img, mean, std):
        img[0] = (img[0] * std[0]) + mean[0]
        img[1] = (img[1] * std[1]) + mean[1]
        img[2] = (img[2] * std[2]) + mean[2]
        img = img * 255

        img = np.clip(img, 0, 255)
        return img

    def get_label_paths(train_path="../input/data_road/data_road/training"):
        label_paths = {re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
                       for path in glob(os.path.join(train_path, 'gt_image_2', '*_road_*.png'))}

        return label_paths

    def get_test_paths(test_path="../input/data_road/data_road/testing"):
        test_paths = [os.path.basename(path)
                          for path in glob(os.path.join(test_path, 'image_2', '*.png'))]

        return test_paths

    def make_layers(cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v

        return nn.Sequential(*layers)

    def gen_test_output(n_class):
        model.eval();
        for i, data in enumerate(testloader):
            sample = data
            images = sample['image']
            images = images.float()
            images = Variable(images.cuda())

            output = model(images)
            output = torch.sigmoid(output)
            output = output.cpu()
            N, c, h, w = output.shape
            pred = np.squeeze(output.detach().cpu().numpy(), axis=0)

            pred = pred.transpose((1, 2, 0))
            pred = pred.argmax(axis=2)
            pred = (pred > 0.5)

            pred = pred.reshape(*pred.shape, 1)
            pred = np.concatenate((pred, np.invert(pred)), axis=2).astype('float')
            pred = np.concatenate((pred, np.zeros((*pred[:,:,0].shape, 1))), axis=2).astype('float')

            pred[pred == 1.0] = 127.0
            images = images.cpu().detach().numpy()
            images = np.squeeze(images)
            images = images.transpose((1, 2, 0))

            images = denormalize(images, mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            output = cv2.addWeighted(images, 0.6, pred, 0.4, 0, dtype = 0)
            output = output/127.0
            output = np.clip(output, 0.0, 1.0)
            yield test_paths[i], output

    def save_inference_samples(output_dir):
        print('Training Finished. Saving test images to: {}'.format(output_dir))
        image_outputs = gen_test_output(n_class=2)
        for name, image in image_outputs:
            plt.imsave(os.path.join(output_dir, name), image)
