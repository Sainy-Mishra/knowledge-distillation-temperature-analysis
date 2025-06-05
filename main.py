import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from models import TeacherNet, StudentNet
from utils import train_teacher, train_student, test_model, knowledge_distillation
from plot_utils import plot_training_curves

def main():
    parser = argparse.ArgumentParser(description='Knowledge Distillation on CIFAR-10')
    parser.add_argument('--mode', choices=['teacher', 'student', 'distill'], default='distill',
                        help='Training mode: teacher, student, or distill')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--temperature', type=float, default=4.0, help='Temperature for knowledge distillation')
    parser.add_argument('--alpha', type=float, default=0.3, help='Weight for distillation loss')
    parser.add_argument('--teacher_path', type=str, default='models/teacher_model.pth', 
                        help='Path to teacher model')
    parser.add_argument('--student_path', type=str, default='models/student_model.pth', 
                        help='Path to save/load student model')

    args = parser.parse_args()

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Data transformations
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Load CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                                 download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, 
                                                transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Initialize models
    teacher = TeacherNet().to(device)
    student = StudentNet().to(device)

    # Create models directory
    os.makedirs('models', exist_ok=True)

    if args.mode == 'teacher':
        print("Training Teacher Model...")
        teacher_losses, teacher_accuracies = train_teacher(teacher, train_loader, test_loader, 
                                                          device, args.epochs, args.lr)
        torch.save(teacher.state_dict(), args.teacher_path)
        plot_training_curves(teacher_losses, teacher_accuracies, 'Teacher Model Training')

    elif args.mode == 'student':
        print("Training Student Model (without distillation)...")
        student_losses, student_accuracies = train_student(student, train_loader, test_loader, 
                                                          device, args.epochs, args.lr)
        torch.save(student.state_dict(), args.student_path)
        plot_training_curves(student_losses, student_accuracies, 'Student Model Training')

    elif args.mode == 'distill':
        print("Knowledge Distillation Training...")

        # Load pre-trained teacher
        if os.path.exists(args.teacher_path):
            teacher.load_state_dict(torch.load(args.teacher_path, map_location=device))
            print("Loaded pre-trained teacher model")
        else:
            print("Training teacher model first...")
            teacher_losses, teacher_accuracies = train_teacher(teacher, train_loader, test_loader, 
                                                              device, args.epochs, args.lr)
            torch.save(teacher.state_dict(), args.teacher_path)

        # Knowledge distillation
        distill_losses, distill_accuracies = knowledge_distillation(
            teacher, student, train_loader, test_loader, device, 
            args.epochs, args.lr, args.temperature, args.alpha
        )
        torch.save(student.state_dict(), args.student_path.replace('.pth', '_distilled.pth'))
        plot_training_curves(distill_losses, distill_accuracies, 'Knowledge Distillation Training')

    # Final evaluation
    teacher_acc = test_model(teacher, test_loader, device)
    student_acc = test_model(student, test_loader, device)

    print(f"\nFinal Results:")
    print(f"Teacher Accuracy: {teacher_acc:.2f}%")
    print(f"Student Accuracy: {student_acc:.2f}%")

if __name__ == '__main__':
    main()