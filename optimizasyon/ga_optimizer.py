import random
import torch
import numpy as np
from pathlib import Path
from models import get_model
from dataset import create_dataloaders, prepare_and_split_data
from train import Trainer

def initialize_individual():
    return {
        'learning_rate': random.uniform(1e-5, 1e-3),
        'batch_size': random.choice([16, 32, 64]),
        'weight_decay': random.uniform(1e-6, 1e-3),
        'l1_lambda': random.uniform(0.0, 1e-3),
        'dropout_rate': random.uniform(0.1, 0.5),
        'beta1': random.uniform(0.85, 0.99),
        'beta2': random.uniform(0.990, 0.999)
    }

def crossover(parent1, parent2):
    child1, child2 = {}, {}
    for key in parent1.keys():
        if random.random() > 0.5:
            child1[key] = parent1[key]
            child2[key] = parent2[key]
        else:
            child1[key] = parent2[key]
            child2[key] = parent1[key]
    return child1, child2

def mutate(individual, mutation_rate=0.2):
    mutated = individual.copy()
    if random.random() < mutation_rate:
        mutated['learning_rate'] = random.uniform(1e-5, 1e-3)
    if random.random() < mutation_rate:
        mutated['batch_size'] = random.choice([16, 32, 64])
    if random.random() < mutation_rate:
        mutated['weight_decay'] = random.uniform(1e-6, 1e-3)
    if random.random() < mutation_rate:
        mutated['l1_lambda'] = random.uniform(0.0, 1e-3)
    if random.random() < mutation_rate:
        mutated['dropout_rate'] = random.uniform(0.1, 0.5)
    if random.random() < mutation_rate:
        mutated['beta1'] = random.uniform(0.85, 0.99)
    if random.random() < mutation_rate:
        mutated['beta2'] = random.uniform(0.990, 0.999)
    return mutated

def evaluate_fitness(individual, config):
    train_paths, train_labels, val_paths, val_labels, _, _ = prepare_and_split_data(
        csv_path=config['csv_path'], val_size=0.15, test_size=0.15, random_state=config['seed']
    )

    train_loader, val_loader, data_info = create_dataloaders(
        train_image_paths=train_paths, train_labels=train_labels,
        val_image_paths=val_paths, val_labels=val_labels,
        batch_size=individual['batch_size'], num_workers=6,
        image_size=224, use_weighted_sampler=False
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = get_model('densenet', num_classes=2, model_size='121', pretrained=True, dropout_rate=individual['dropout_rate'])

    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        num_classes=2, class_weights=data_info.get('class_weights', None),
        device=device, learning_rate=individual['learning_rate'],
        weight_decay=individual['weight_decay'], output_dir=config['output_dir'],
        beta1=individual['beta1'], beta2=individual['beta2'], l1_lambda=individual['l1_lambda']
    )

    history = trainer.train(num_epochs=config['eval_epochs'], save_best=False)
    val_f1_macro = history['val_f1'][-1] if len(history['val_f1']) > 0 else 0.0
    return val_f1_macro

def main():
    CONFIG = {
        'csv_path': '/mnt/DEPO/Birads_Tubitak/kodlar/okul_etiketleri.csv',
        'output_dir': 'ga_outputs',
        'seed': 42,
        'eval_epochs': 5, 
        'population_size': 10,
        'generations': 5
    }

    Path(CONFIG['output_dir']).mkdir(parents=True, exist_ok=True)
    
    torch.manual_seed(CONFIG['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(CONFIG['seed'])

    population = [initialize_individual() for _ in range(CONFIG['population_size'])]
    best_overall_individual = None
    best_overall_fitness = -1.0

    for gen in range(CONFIG['generations']):
        print(f"\n--- Jenerasyon {gen + 1}/{CONFIG['generations']} ---")
        fitness_scores = []
        for idx, ind in enumerate(population):
            fitness = evaluate_fitness(ind, CONFIG)
            fitness_scores.append((ind, fitness))
            
            if fitness > best_overall_fitness:
                best_overall_fitness = fitness
                best_overall_individual = ind

        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        population = [x[0] for x in fitness_scores[:CONFIG['population_size'] // 2]]

        next_generation = []
        while len(next_generation) < CONFIG['population_size']:
            p1, p2 = random.sample(population, 2)
            c1, c2 = crossover(p1, p2)
            next_generation.extend([mutate(c1), mutate(c2)])
            
        population = next_generation[:CONFIG['population_size']]

    print("\nOptimizasyon Tamamlandi.")
    print("En Iyi Parametreler:")
    for k, v in best_overall_individual.items():
        print(f"  {k}: {v}")
    print(f"En Iyi F1 Skoru: {best_overall_fitness}")

if __name__ == "__main__":
    main()
