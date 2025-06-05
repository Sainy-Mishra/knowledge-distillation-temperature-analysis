import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm

def train_teacher(model, train_loader, test_loader, device, epochs=20, lr=0.001):
    """
    Train the teacher model using standard supervised learning
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    train_losses = []
    test_accuracies = []

    print("Training Teacher Model...")
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        # Training phase
        model.train()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch_idx, (data, targets) in enumerate(pbar):
            data, targets = data.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % 50 == 0:
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })

        scheduler.step()

        # Evaluation phase
        test_acc = test_model(model, test_loader, device, verbose=False)
        avg_loss = running_loss / len(train_loader)

        train_losses.append(avg_loss)
        test_accuracies.append(test_acc)

        print(f'Epoch {epoch+1}: Loss: {avg_loss:.4f}, Test Acc: {test_acc:.2f}%')

    return train_losses, test_accuracies

def train_student(model, train_loader, test_loader, device, epochs=20, lr=0.001):
    """
    Train the student model using standard supervised learning (without distillation)
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    train_losses = []
    test_accuracies = []

    print("Training Student Model (without distillation)...")
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        model.train()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch_idx, (data, targets) in enumerate(pbar):
            data, targets = data.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % 50 == 0:
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })

        scheduler.step()

        test_acc = test_model(model, test_loader, device, verbose=False)
        avg_loss = running_loss / len(train_loader)

        train_losses.append(avg_loss)
        test_accuracies.append(test_acc)

        print(f'Epoch {epoch+1}: Loss: {avg_loss:.4f}, Test Acc: {test_acc:.2f}%')

    return train_losses, test_accuracies

def knowledge_distillation(teacher, student, train_loader, test_loader, device, 
                          epochs=20, lr=0.001, temperature=4.0, alpha=0.3):
    """
    Perform knowledge distillation from teacher to student

    Args:
        teacher: Pre-trained teacher model
        student: Student model to be trained
        temperature: Temperature for softmax (higher = softer probabilities)
        alpha: Weight for distillation loss vs hard target loss
    """
    teacher.eval()  # Teacher in evaluation mode
    student.train()

    # Loss functions
    criterion_ce = nn.CrossEntropyLoss()  # For hard targets
    criterion_kd = nn.KLDivLoss(reduction='batchmean')  # For soft targets

    optimizer = optim.Adam(student.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    train_losses = []
    test_accuracies = []

    print(f"Knowledge Distillation (T={temperature}, α={alpha})...")

    for epoch in range(epochs):
        running_loss = 0.0
        running_ce_loss = 0.0
        running_kd_loss = 0.0
        correct = 0
        total = 0

        student.train()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')

        for batch_idx, (data, targets) in enumerate(pbar):
            data, targets = data.to(device), targets.to(device)

            optimizer.zero_grad()

            # Forward pass through both models
            with torch.no_grad():
                teacher_logits = teacher(data)

            student_logits = student(data)

            # Compute losses
            # 1. Standard cross-entropy loss with hard targets
            ce_loss = criterion_ce(student_logits, targets)

            # 2. Knowledge distillation loss with soft targets
            teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
            student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
            kd_loss = criterion_kd(student_log_probs, teacher_probs) * (temperature ** 2)

            # 3. Combined loss
            loss = alpha * kd_loss + (1 - alpha) * ce_loss

            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item()
            running_ce_loss += ce_loss.item()
            running_kd_loss += kd_loss.item()

            _, predicted = student_logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % 50 == 0:
                pbar.set_postfix({
                    'Total': f'{loss.item():.4f}',
                    'CE': f'{ce_loss.item():.4f}',
                    'KD': f'{kd_loss.item():.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })

        scheduler.step()

        test_acc = test_model(student, test_loader, device, verbose=False)
        avg_loss = running_loss / len(train_loader)
        avg_ce_loss = running_ce_loss / len(train_loader)
        avg_kd_loss = running_kd_loss / len(train_loader)

        train_losses.append(avg_loss)
        test_accuracies.append(test_acc)

        print(f'Epoch {epoch+1}: Total: {avg_loss:.4f}, CE: {avg_ce_loss:.4f}, '
              f'KD: {avg_kd_loss:.4f}, Test Acc: {test_acc:.2f}%')

    return train_losses, test_accuracies

def test_model(model, test_loader, device, verbose=True):
    """
    Evaluate model on test set
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    if verbose:
        print(f'Test Accuracy: {accuracy:.2f}% ({correct}/{total})')

    return accuracy

def distillation_loss(student_logits, teacher_logits, targets, temperature, alpha):
    """
    Compute the knowledge distillation loss

    Args:
        student_logits: Raw outputs from student model
        teacher_logits: Raw outputs from teacher model  
        targets: Ground truth labels
        temperature: Temperature for softmax
        alpha: Weight for distillation loss

    Returns:
        Combined loss value
    """
    # Hard target loss
    ce_loss = F.cross_entropy(student_logits, targets)

    # Soft target loss
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    kd_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)

    # Combined loss
    total_loss = alpha * kd_loss + (1 - alpha) * ce_loss

    return total_loss, ce_loss, kd_loss

def compare_models(teacher, student, test_loader, device):
    """
    Compare teacher and student model performance
    """
    print("\n" + "="*50)
    print("MODEL COMPARISON")
    print("="*50)

    teacher_acc = test_model(teacher, test_loader, device, verbose=False)
    student_acc = test_model(student, test_loader, device, verbose=False)

    print(f"Teacher Accuracy: {teacher_acc:.2f}%")
    print(f"Student Accuracy: {student_acc:.2f}%")
    print(f"Performance Gap: {teacher_acc - student_acc:.2f}%")

    # Model size comparison
    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student.parameters())

    print(f"\nTeacher Parameters: {teacher_params:,}")
    print(f"Student Parameters: {student_params:,}")
    print(f"Compression Ratio: {teacher_params / student_params:.2f}x")

    return teacher_acc, student_acc