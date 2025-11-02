from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPVisionModel
from medclip import MedCLIPProcessor
from medclip import PromptClassifier

processor = MedCLIPProcessor()
model2 = MedCLIPModel(vision_cls=MedCLIPVisionModel)
model2.from_pretrained()

model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
model.from_pretrained()