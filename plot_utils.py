import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

def plot_training_curves(losses, accuracies, title):
    """
    Plot training loss and test accuracy curves
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot training loss
    ax1.plot(losses, 'b-', linewidth=2, label='Training Loss')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot test accuracy
    ax2.plot(accuracies, 'r-', linewidth=2, label='Test Accuracy')
    ax2.set_title('Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()

def compare_training_curves(teacher_data, student_data, distill_data):
    """
    Compare training curves for teacher, student, and distillation
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot losses
    if teacher_data:
        ax1.plot(teacher_data[0], 'b-', linewidth=2, label='Teacher')
    if student_data:
        ax1.plot(student_data[0], 'r-', linewidth=2, label='Student (No Distillation)')
    if distill_data:
        ax1.plot(distill_data[0], 'g-', linewidth=2, label='Student (With Distillation)')

    ax1.set_title('Training Loss Comparison')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot accuracies
    if teacher_data:
        ax2.plot(teacher_data[1], 'b-', linewidth=2, label='Teacher')
    if student_data:
        ax2.plot(student_data[1], 'r-', linewidth=2, label='Student (No Distillation)')
    if distill_data:
        ax2.plot(distill_data[1], 'g-', linewidth=2, label='Student (With Distillation)')

    ax2.set_title('Test Accuracy Comparison')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('training_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_temperature_analysis(temperatures, accuracies):
    """
    Plot the effect of temperature on distillation performance
    """
    plt.figure(figsize=(10, 6))
    plt.plot(temperatures, accuracies, 'bo-', linewidth=2, markersize=8)
    plt.title('Effect of Temperature on Knowledge Distillation', fontsize=14, fontweight='bold')
    plt.xlabel('Temperature', fontsize=12)
    plt.ylabel('Student Test Accuracy (%)', fontsize=12)
    plt.grid(True, alpha=0.3)

    # Annotate best temperature
    best_idx = np.argmax(accuracies)
    best_temp = temperatures[best_idx]
    best_acc = accuracies[best_idx]
    plt.annotate(f'Best: T={best_temp}, Acc={best_acc:.2f}%', 
                xy=(best_temp, best_acc), xytext=(best_temp+1, best_acc+1),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')

    plt.tight_layout()
    plt.savefig('temperature_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_soft_targets(teacher_logits, student_logits, temperature, class_names=None):
    """
    Visualize soft targets from teacher and student models
    """
    if class_names is None:
        class_names = [f'Class {i}' for i in range(teacher_logits.shape[1])]

    # Compute softmax probabilities
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    student_probs = F.softmax(student_logits / temperature, dim=1)

    # Plot first sample
    sample_idx = 0
    teacher_sample = teacher_probs[sample_idx].cpu().numpy()
    student_sample = student_probs[sample_idx].cpu().numpy()

    x = np.arange(len(class_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, teacher_sample, width, label='Teacher', alpha=0.8)
    rects2 = ax.bar(x + width/2, student_sample, width, label='Student', alpha=0.8)

    ax.set_xlabel('Classes')
    ax.set_ylabel('Probability')
    ax.set_title(f'Soft Targets Comparison (Temperature = {temperature})')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('soft_targets_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_model_architecture_comparison():
    """
    Create a visual comparison of teacher vs student architecture
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))

    # Teacher architecture
    teacher_layers = ['Input\n(3x32x32)', 'Conv+BN\n(64)', 'ResBlock×2\n(64)', 
                     'ResBlock×2\n(128)', 'ResBlock×2\n(256)', 'ResBlock×2\n(512)',
                     'Global Pool', 'FC\n(10)']
    teacher_params = [0, 1728, 147968, 525312, 2099200, 8390656, 0, 5130]

    y_pos = np.arange(len(teacher_layers))
    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(teacher_layers)))

    ax1.barh(y_pos, [p/1000 for p in teacher_params], color=colors)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(teacher_layers)
    ax1.set_xlabel('Parameters (K)')
    ax1.set_title('Teacher Network\n(~11.2M parameters)')
    ax1.grid(True, alpha=0.3)

    # Student architecture
    student_layers = ['Input\n(3x32x32)', 'Conv+BN\n(32)', 'Conv+BN\n(64)', 
                     'Conv+BN\n(128)', 'FC\n(256)', 'FC\n(10)']
    student_params = [0, 896, 18496, 73856, 524544, 2570]

    y_pos = np.arange(len(student_layers))
    colors = plt.cm.Reds(np.linspace(0.4, 0.8, len(student_layers)))

    ax2.barh(y_pos, [p/1000 for p in student_params], color=colors)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(student_layers)
    ax2.set_xlabel('Parameters (K)')
    ax2.set_title('Student Network\n(~620K parameters)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('architecture_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_knowledge_distillation_diagram():
    """
    Create a diagram explaining the knowledge distillation process
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    # Hide axes
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Teacher model
    teacher_box = plt.Rectangle((0.5, 5), 2, 2, fill=True, color='lightblue', 
                               edgecolor='blue', linewidth=2)
    ax.add_patch(teacher_box)
    ax.text(1.5, 6, 'Teacher\nModel\n(Large)', ha='center', va='center', 
            fontsize=12, fontweight='bold')

    # Student model
    student_box = plt.Rectangle((0.5, 1), 2, 2, fill=True, color='lightcoral', 
                               edgecolor='red', linewidth=2)
    ax.add_patch(student_box)
    ax.text(1.5, 2, 'Student\nModel\n(Small)', ha='center', va='center', 
            fontsize=12, fontweight='bold')

    # Input
    input_box = plt.Rectangle((0.5, 3.25), 2, 1.5, fill=True, color='lightgreen', 
                             edgecolor='green', linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.5, 4, 'Input\nImage', ha='center', va='center', 
            fontsize=12, fontweight='bold')

    # Soft targets
    soft_box = plt.Rectangle((4, 5.5), 2.5, 1, fill=True, color='lightyellow', 
                            edgecolor='orange', linewidth=2)
    ax.add_patch(soft_box)
    ax.text(5.25, 6, 'Soft Targets\n(Temperature Scaling)', ha='center', va='center', 
            fontsize=10, fontweight='bold')

    # Hard targets
    hard_box = plt.Rectangle((4, 4), 2.5, 1, fill=True, color='lightgray', 
                            edgecolor='black', linewidth=2)
    ax.add_patch(hard_box)
    ax.text(5.25, 4.5, 'Hard Targets\n(Ground Truth)', ha='center', va='center', 
            fontsize=10, fontweight='bold')

    # Loss function
    loss_box = plt.Rectangle((7.5, 2), 2, 3, fill=True, color='plum', 
                            edgecolor='purple', linewidth=2)
    ax.add_patch(loss_box)
    ax.text(8.5, 3.5, 'Combined\nLoss\n\nα × KD Loss\n+\n(1-α) × CE Loss', 
            ha='center', va='center', fontsize=10, fontweight='bold')

    # Arrows
    # Input to models
    ax.arrow(2.6, 4, 0, 2.5, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax.arrow(2.6, 4, 0, -1.5, head_width=0.1, head_length=0.1, fc='black', ec='black')

    # Teacher to soft targets
    ax.arrow(2.6, 6, 1.3, 0, head_width=0.1, head_length=0.1, fc='orange', ec='orange')

    # Hard targets to loss
    ax.arrow(6.6, 4.5, 0.8, -1.5, head_width=0.1, head_length=0.1, fc='black', ec='black')

    # Soft targets to loss
    ax.arrow(6.6, 6, 0.8, -2.5, head_width=0.1, head_length=0.1, fc='orange', ec='orange')

    # Student to loss
    ax.arrow(2.6, 2, 4.8, 1, head_width=0.1, head_length=0.1, fc='red', ec='red')

    # Title
    ax.text(5, 7.5, 'Knowledge Distillation Process', ha='center', va='center', 
            fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig('knowledge_distillation_diagram.png', dpi=300, bbox_inches='tight')
    plt.show()

# CIFAR-10 class names
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']