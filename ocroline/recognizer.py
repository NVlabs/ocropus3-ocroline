from numpy import *
import torch
from torch.autograd import Variable
from ocroline import lineest
from dltrainers import helpers
from dlinputs import sequence

class LineRecognizer(object):
    def __init__(self, mname, codec=None):
        self.codec = codec or sequence.ascii_codec
        self.model = torch.load("/home/tmb/tmb-models/line2-000003330-004377.pt")
        self.model.cuda()
        self.model.eval()
        self.normalizer = lineest.CenterNormalizer()
    def recognize_line(self, line_image):
        assert amin(line_image) >= 0
        assert amax(line_image) <= 1
        self.line_image = array(line_image > 0.5, 'f')
        self.normalized = self.normalizer.measure_and_normalize(self.line_image)
        tinput = torch.FloatTensor(self.normalized).cuda()[None, :, :, None]
        output = self.model.forward(Variable(tinput)).data.cpu()
        self.probs = array(helpers.sequence_softmax(output), 'f')
        return self.codec.decode_batch(self.probs)[0]
    def recognize_batch(self, line_images):
        images = []
        for image in line_images:
            assert amin(image) >= 0
            assert amax(image) <= 1
            image = array(image > 0.5, 'f')
            normalized = self.normalizer.measure_and_normalize(image)
            images.append(normalized)
        self.batch = sequence.seq_makebatch(images)
        tinput = torch.FloatTensor(self.batch).cuda()[:, :, :, None]
        output = self.model.forward(Variable(tinput)).data.cpu()
        self.probs = array(helpers.sequence_softmax(output), 'f')
        return self.codec.decode_batch(self.probs)
