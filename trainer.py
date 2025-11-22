#!/usr/bin/env python3
"""
MAMARDASHVILI MEGA-MODEL - обучается делать всё: код, автоматизация, креатив
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset, load_dataset
import json
from typing import Dict, List, Any
import os

class UniversalArchitecture(nn.Module):
    """Универсальная архитектура для любых задач"""
    
    def __init__(self, base_model_name="microsoft/DialoGPT-large"):
        super().__init__()
        
        # Квантование для экономии памяти
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        # Подготовка для PEFT
        self.model = prepare_model_for_kbit_training(self.model)
        
        # LoRA конфиг
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        )
        
        self.model = get_peft_model(self.model, peft_config)
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    def generate(self, prompt: str, **kwargs):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=kwargs.get('max_length', 512),
                temperature=kwargs.get('temperature', 0.7),
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

class MegaDataset:
    """Мега-датасет для обучения на всём подряд"""
    
    def __init__(self):
        self.datasets = {}
        
    def load_code_dataset(self):
        """Датасет программирования"""
        try:
            code_dataset = load_dataset("bigcode/the-stack", data_dir="data/python", split="train[:1%]")
            formatted = []
            for item in code_dataset:
                formatted.append(f"<code>\n{item['content']}\n</code>")
            return formatted
        except:
            # Fallback локальные данные
            return self._create_fallback_code_data()
    
    def load_creative_dataset(self):
        """Креативные тексты"""
        creative_prompts = [
            "Напиши поэму о искусственном интеллекте",
            "Придумай диалог между двумя философами",
            "Опиши будущее через 100 лет",
            "Напиши техническое описание квантового компьютера",
            "Создай бизнес-план для стартапа в сфере AI"
        ]
        return creative_prompts
    
    def load_automation_dataset(self):
        """Данные для автоматизации"""
        automation_data = [
            "Автоматизируй процесс обработки данных: шаг 1 - сбор, шаг 2 - очистка, шаг 3 - анализ",
            "Создай скрипт для автоматического деплоя приложения",
            "Напиши программу для мониторинга системных ресурсов",
            "Автоматизируй процесс бэкапа базы данных",
            "Создай систему для автоматического тестирования кода"
        ]
        return automation_data
    
    def build_mega_dataset(self):
        """Собираем все данные вместе"""
        all_data = []
        
        # Код
        code_data = self.load_code_dataset()
        all_data.extend(code_data)
        
        # Креатив
        creative_data = self.load_creative_dataset()
        all_data.extend(creative_data)
        
        # Автоматизация
        auto_data = self.load_automation_dataset()
        all_data.extend(auto_data)
        
        return Dataset.from_dict({"text": all_data})
    
    def _create_fallback_code_data(self):
        """Локальные данные программирования"""
        return [
            "def calculate_fibonacci(n):\n    if n <= 1:\n        return n\n    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)",
            "class NeuralNetwork:\n    def __init__(self, layers):\n        self.layers = layers\n    def forward(self, x):\n        for layer in self.layers:\n            x = layer(x)\n        return x",
            "async def process_data(data):\n    results = []\n    async for item in data:\n        processed = await clean_data(item)\n        results.append(processed)\n    return results"
        ]

class MamardashviliTrainer:
    """Тренер мега-модели"""
    
    def __init__(self):
        self.model = UniversalArchitecture()
        self.dataset_builder = MegaDataset()
        
    def train(self, output_dir: str = "./mamardashvili-mega-model"):
        """Запуск обучения"""
        
        # Сбор данных
        print("📦 Собираю мега-датасет...")
        dataset = self.dataset_builder.build_mega_dataset()
        
        # Токенизация
        def tokenize_function(examples):
            return self.model.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=1024,
                return_tensors="pt"
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # Аргументы обучения
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            logging_steps=10,
            save_steps=500,
            learning_rate=2e-4,
            fp16=True,
            optim="adamw_torch",
            report_to=None,
            ddp_find_unused_parameters=False
        )
        
        # Тренер
        trainer = Trainer(
            model=self.model.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.model.tokenizer,
                mlm=False
            )
        )
        
        # Запуск обучения
        print("🚀 Начинаю обучение мега-модели...")
        trainer.train()
        
        # Сохранение
        trainer.save_model()
        self.model.tokenizer.save_pretrained(output_dir)
        
        print(f"✅ Модель сохранена в {output_dir}")
        
    def generate_code(self, task: str) -> str:
        """Генерация кода"""
        prompt = f"<code>\n# Задача: {task}\n# Решение:"
        return self.model.generate(prompt, max_length=500)
    
    def generate_automation(self, process: str) -> str:
        """Генерация автоматизации"""
        prompt = f"Автоматизируй процесс: {process}\n\nРешение:"
        return self.model.generate(prompt, max_length=400)
    
    def generate_creative(self, theme: str) -> str:
        """Креативная генерация"""
        prompt = f"Тема: {theme}\n\nТекст:"
        return self.model.generate(prompt, max_length=300)

# ИНТЕРФЕЙС
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Запустить обучение")
    parser.add_argument("--generate", type=str, help="Текст для генерации")
    parser.add_argument("--mode", choices=["code", "auto", "creative"], help="Режим генерации")
    
    args = parser.parse_args()
    
    trainer = MamardashviliTrainer()
    
    if args.train:
        trainer.train()
    elif args.generate and args.mode:
        if args.mode == "code":
            result = trainer.generate_code(args.generate)
        elif args.mode == "auto":
            result = trainer.generate_automation(args.generate)
        else:
            result = trainer.generate_creative(args.generate)
        print(result)
