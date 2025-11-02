from medclip import MedCLIPModel, MedCLIPVisionModelViT
from medclip import MedCLIPProcessor
from medclip import PromptClassifier

processor = MedCLIPProcessor()
model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
model2 = MedCLIPModel(vision_cls=MedCLIPVisionModel)
model.from_pretrained()
model2.from_pretrained()