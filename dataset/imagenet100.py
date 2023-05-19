from torchvision.datasets import ImageFolder, ImageNet
import os
import numpy as np

class imagenet100(ImageNet):

    imagenet_templates = [
        'a photo of a {}.',
        'a bad photo of a {}.',
        'a photo of many {}.',
        'a sculpture of a {}.',
        'a photo of the hard to see {}.',
        'a low resolution photo of the {}.',
        'a rendering of a {}.',
        'graffiti of a {}.',
        'a bad photo of the {}.',
        'a cropped photo of the {}.',
        'a tattoo of a {}.',
        'the embroidered {}.',
        'a photo of a hard to see {}.',
        'a bright photo of a {}.',
        'a photo of a clean {}.',
        'a photo of a dirty {}.',
        'a dark photo of the {}.',
        'a drawing of a {}.',
        'a photo of my {}.',
        'the plastic {}.',
        'a photo of the cool {}.',
        'a close-up photo of a {}.',
        'a black and white photo of the {}.',
        'a painting of the {}.',
        'a painting of a {}.',
        'a pixelated photo of the {}.',
        'a sculpture of the {}.',
        'a bright photo of the {}.',
        'a cropped photo of a {}.',
        'a plastic {}.',
        'a photo of the dirty {}.',
        'a jpeg corrupted photo of a {}.',
        'a blurry photo of the {}.',
        'a photo of the {}.',
        'a good photo of the {}.',
        'a rendering of the {}.',
        'a {} in a video game.',
        'a photo of one {}.',
        'a doodle of a {}.',
        'a close-up photo of the {}.',
        'the origami {}.',
        'the {} in a video game.',
        'a sketch of a {}.',
        'a doodle of the {}.',
        'a origami {}.',
        'a low resolution photo of a {}.',
        'the toy {}.',
        'a rendition of the {}.',
        'a photo of the clean {}.',
        'a photo of a large {}.',
        'a rendition of a {}.',
        'a photo of a nice {}.',
        'a photo of a weird {}.',
        'a blurry photo of a {}.',
        'a cartoon {}.',
        'art of a {}.',
        'a sketch of the {}.',
        'a embroidered {}.',
        'a pixelated photo of a {}.',
        'itap of the {}.',
        'a jpeg corrupted photo of the {}.',
        'a good photo of a {}.',
        'a plushie {}.',
        'a photo of the nice {}.',
        'a photo of the small {}.',
        'a photo of the weird {}.',
        'the cartoon {}.',
        'art of the {}.',
        'a drawing of the {}.',
        'a photo of the large {}.',
        'a black and white photo of a {}.',
        'the plushie {}.',
        'a dark photo of a {}.',
        'itap of a {}.',
        'graffiti of the {}.',
        'a toy {}.',
        'itap of my {}.',
        'a photo of a cool {}.',
        'a photo of a small {}.',
        'a tattoo of the {}.',
        ]
        
        
    new_classes = [
        'American robin', 'Gila monster', 'eastern hog-nosed snake', 'garter snake', 'green mamba', 
        'European garden spider', 'lorikeet', 'goose', 'rock crab', 'fiddler crab', 'American lobster', 
        'little blue heron', 'American coot', 'Chihuahua', 'Shih Tzu', 'Papillon', 'toy terrier', 
        'Treeing Walker Coonhound', 'English foxhound', 'borzoi', 'Saluki', 'American Staffordshire Terrier', 
        'Chesapeake Bay Retriever', 'Vizsla', 'Kuvasz', 'Komondor', 'Rottweiler', 'Dobermann', 'Boxer', 
        'Great Dane', 'Standard Poodle', 'Mexican hairless dog (xoloitzcuintli)', 'coyote', 'African wild dog', 
        'red fox','tabby cat', 'meerkat', 'dung beetle', 'stick insect', 'leafhopper', 'hare', 'wild boar', 
        'gibbon', 'langur', 'ambulance', 'baluster handrail', 'bassinet', 'boathouse', 'poke bonnet', 
        'bottle cap', 'car wheel', 'bell or wind chime', 'movie theater', 'cocktail shaker', 'computer keyboard', 
        'Dutch oven', 'football helmet', 'gas mask or respirator', 'hard disk drive', 'harmonica', 'honeycomb', 
        'clothes iron', 'jeans', 'lampshade', 'laptop computer', 'milk can', 'mixing bowl', 'modem', 'moped', 
        'graduation cap', 'mousetrap', 'obelisk', 'park bench', 'pedestal', 'pickup truck', 'pirate ship', 
        'purse', 'fishing casting reel', 'rocking chair', 'rotisserie', 'safety pin', 'sarong', 'balaclava ski mask', 
        'slide rule', 'stretcher', 'front curtain', 'throne', 'tile roof', 'tripod', 'hot tub', 'vacuum cleaner', 
        'window screen', 'airplane wing', 'cabbage', 'cauliflower', 'pineapple', 'carbonara', 'chocolate syrup', 
        'gyromitra', 'stinkhorn mushroom']

    def __init__(self, root, transform=None,train=True):
        split = 'train' if train else 'val'
        super(imagenet100, self).__init__(os.path.join(root), split=split,transform=transform)
        self.classes = self.new_classes

    def prompts(self,mode='single'):
        if mode == 'single':
            prompts = [[self.imagenet_templates[0].format(label)] for label in self.new_classes]
            return prompts
        elif mode == 'ensemble':
            prompts = [[template.format(label) for template in self.imagenet_templates] for label in self.new_classes]
            return prompts

    def get_labels(self):
        return np.array(self.targets)

    def get_classes(self):
        return self.new_classes