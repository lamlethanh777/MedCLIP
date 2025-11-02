import pdb, os
import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from medclip.modeling_medclip import MedCLIPModel, PromptClassifier, MedCLIPVisionModel, MedCLIPVisionModelViT
from medclip.dataset import ImageTextContrastiveDataset, ZeroShotImageDataset
from medclip.dataset import ImageTextContrastiveCollator, ZeroShotImageCollator
from medclip.losses import ImageTextContrastiveLoss
from medclip.trainer import Trainer
from medclip.evaluator import Evaluator
from medclip import constants
from medclip.prompts import generate_class_prompts, generate_chexpert_class_prompts, generate_covid_class_prompts, generate_rsna_class_prompts
from medclip import utils

# set random seed
utils.set_random_seed(42)

# set cuda devices
os.environ['CUDA_VISIBLE_DEVICES']='0'
device = "cuda:0" if torch.cuda.is_available() else "cpu"

def setup_dataset_config(dataset_name):
    """Configure dataset-specific settings"""
    configs = {
        'chexpert': {
            'datalist': ['chexpert-test'],
            'class_names': constants.CHEXPERT_COMPETITION_TASKS,
            'prompt_generator': generate_chexpert_class_prompts,
            'mode': 'multiclass', # 5x200 does not have multilabel
            'n_prompts': 10,
            'path': None
        },
        'covid': {
            'datalist': ['covid-test'],
            'class_names': constants.COVID_TASKS,
            'prompt_generator': generate_covid_class_prompts,
            'mode': 'binary',  # COVID vs Normal
            'n_prompts': 10,
            'path': None
        },
        'rsna': {
            'datalist': ['rsna-balanced-test'],
            'class_names': constants.RSNA_TASKS,
            'prompt_generator': generate_rsna_class_prompts,
            'mode': 'binary',  # Pneumonia vs Normal
            'n_prompts': 10,
            'path': None
        },
        'mimic': {
            'datalist': ['mimic-5x200'],
            'class_names': constants.CHEXPERT_COMPETITION_TASKS,
            'prompt_generator': generate_chexpert_class_prompts,
            'mode': 'multiclass', # 5x200 does not have multilabel
            'n_prompts': 10,
            'path': None
        },
        'openi': {
            'datalist': ['openi'],
            'class_names': constants.CHEXPERT_COMPETITION_TASKS,
            'prompt_generator': generate_chexpert_class_prompts,
            'mode': 'multiclass', # 5x200 does not have multilabel
            'n_prompts': 10,
            'path': None
        },
    }
    return configs[dataset_name]

def setup_model(model_type='vit', pretrained=True):
    """Setup MedCLIP model with specified vision backbone"""
    if model_type.lower() == 'vit':
        vision_cls = MedCLIPVisionModelViT
        print("Using Vision Transformer (ViT) backbone")
    elif model_type.lower() == 'resnet':
        vision_cls = MedCLIPVisionModel
        print("Using ResNet backbone")
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'vit' or 'resnet'")
    
    model = MedCLIPModel(vision_cls=vision_cls)
    if pretrained:
        model.from_pretrained()
        print("Loaded pretrained weights")
    model.cuda()
    return model

def run_zero_shot_evaluation(dataset_name='chexpert', model_type='vit', ensemble=False, batch_size=256):
    """
    Run zero-shot evaluation on specified dataset
    
    Args:
        dataset_name: 'chexpert', 'covid', 'rsna', or 'mimic'
        model_type: 'vit' or 'resnet'
        batch_size: evaluation batch size
    """
    print(f"\n{'='*60}")
    print(f"Zero-Shot Evaluation: {dataset_name.upper()} dataset")
    print(f"Model: MedCLIP-{model_type.upper()}")
    print(f"{'='*60}\n")
    
    # Get dataset configuration
    config = setup_dataset_config(dataset_name)
    
    # Generate class prompts
    print(f"Generating {config['n_prompts']} prompts per class...")
    cls_prompts = config['prompt_generator'](n=config['n_prompts'])
    print(f"Classes: {config['class_names']}")

    # Setup dataset
    print(f"\nLoading dataset from: {config['datalist']}")
    
    eval_dataset = ZeroShotImageDataset(
        datalist=config['datalist'],
        class_names=config['class_names'],
        path=config['path']
    )
    print(f"Dataset size: {len(eval_dataset)} images")
    
    # Setup collator
    eval_collate_fn = ZeroShotImageCollator(
        cls_prompts=cls_prompts,
        mode=config['mode']
    )
    
    # Setup dataloader
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        collate_fn=eval_collate_fn,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
    )

    # Setup model
    model = setup_model(model_type=model_type, pretrained=True)
    medclip_clf = PromptClassifier(model, ensemble=ensemble)
    
    # Setup evaluator
    evaluator = Evaluator(
        medclip_clf=medclip_clf,
        eval_dataloader=eval_dataloader,
        mode=config['mode'],
    )
    
    # Run evaluation
    print("\nStarting evaluation...")
    results = evaluator.evaluate()
    print("\nDone!")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Zero-Shot Evaluation for MedCLIP')
    parser.add_argument('--dataset', type=str, default='chexpert',
                        choices=['chexpert', 'covid', 'rsna', 'mimic'],
                        help='Dataset to evaluate on')
    parser.add_argument('--model', type=str, default='vit',
                        choices=['vit', 'resnet'],
                        help='MedCLIP model type')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Evaluation batch size')
    parser.add_argument('--all', action='store_true',
                        help='Evaluate on all datasets')
    
    args = parser.parse_args()

    if args.all:
        # Evaluate on all datasets
        datasets = ['chexpert', 'covid', 'rsna', 'mimic']
        all_results = {}
        for dataset in datasets:
            try:
                results = run_zero_shot_evaluation(
                    dataset_name=dataset,
                    model_type=args.model,
                    batch_size=args.batch_size
                )
                all_results[dataset] = results
            except Exception as e:
                print(f"Error evaluating {dataset}: {e}")
                continue
        
        # Print summary
        print("\n" + "="*60)
        print("SUMMARY - All Datasets")
        print("="*60)
        for dataset, results in all_results.items():
            print(f"\n{dataset.upper()}:")
            for metric, value in results.items():
                print(f"  {metric}: {value:.4f}")
    else:
        # Evaluate on single dataset
        run_zero_shot_evaluation(
            dataset_name=args.dataset,
            model_type=args.model,
            batch_size=args.batch_size
        )

if __name__ == '__main__':
    main()
