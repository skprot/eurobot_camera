import model
from utils import *
import torch
import time


class Detect():

    def __init__(self, config_path, weights_path, classes, confidence=0.4, DIM=[2448, 2048]):
        self.config = config_path
        self.yolo = model.Model(self.config)
        self.yolo.load_weights(weights_path)
        self.classes = classes
        self.confidence = confidence
        self.nms_thesh = 0.4
        self.num_classes = len(classes)
        self.inp_dim = int(self.yolo.net_info["height"])
        self.im_dim = DIM
        self.im_dim = torch.FloatTensor(self.im_dim).repeat(1, 2)
        self.CUDA = torch.cuda.is_available()

        if self.CUDA:
            self.im_dim = self.im_dim.cuda()
            self.yolo.cuda()
        else:
            print('CUDA is fasle! Using CPU \n')

    def detect(self, frame):
        start = time.time()
        img, new_h = model.prep_image(frame, self.inp_dim)

        if self.CUDA:
            img = img.cuda()

        output = self.yolo(img, self.CUDA)
        output = write_results(output, self.confidence, self.num_classes, nms_conf=self.nms_thesh)

        try:
            output[:, 1:3] = torch.clamp(output[:, 1:3], 0.0, float(self.inp_dim))
            output[:, 3:5] = torch.clamp(output[:, 3:5], 0.0, float(new_h))

            im_dim = self.im_dim.repeat(output.size(0), 1)

            scaling_factor = torch.min(416 / im_dim, 1)[0].view(-1, 1)

            output[:, [1, 3]] -= (self.inp_dim - scaling_factor * im_dim[:, 0].view(-1, 1)) / 2
            output[:, [2, 4]] -= (self.inp_dim - scaling_factor * im_dim[:, 1].view(-1, 1)) / 2

            output[:, 1:5] /= scaling_factor

            for i in range(output.shape[0]):
                output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim[i, 0])
                output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim[i, 1])

            # TODO make parser and make strings

            #list(map(lambda x: model.write(x, frame, self.classes), output))  # FIXME remove then

            return output.shape[0]

        except:
            pass
            return 0
