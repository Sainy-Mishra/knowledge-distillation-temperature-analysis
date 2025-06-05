# 📘 Knowledge Distillation Temperature Analysis

This project explores the effect of **temperature scaling** in **knowledge distillation** using the CIFAR-10 dataset. A large **Teacher model** is used to train a smaller, faster **Student model**. By varying the **temperature parameter**, we analyze how "softening" the output predictions affects the student’s learning performance.

---

## 🎯 Project Goals

- Implement a teacher-student knowledge distillation framework.
- Run experiments with different temperature values.
- Evaluate the impact of temperature on student accuracy.
- Visualize and compare results.

---

## 🧠 What is Knowledge Distillation?

**Knowledge Distillation** is a model compression technique where a small model (Student) learns to mimic a larger, well-trained model (Teacher). Instead of learning just from ground-truth labels, the student also learns from the **soft predictions** of the teacher, which include richer information about class relationships.

The **temperature** controls how soft the teacher’s predictions are. Higher temperatures spread the probabilities more evenly across classes, helping the student learn more subtle patterns.

---

## 🏗️ Project Structure

```bash
project/
│
├── main.py
├── utils.py
├── models.py
├── train.py
├── README.md
└── requirements.txt
```

---

## 🖥️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Sainy-Mishra/knowledge-distillation-temperature-analysis.git
cd knowledge-distillation-temperature-analysis

```
### 2. Install requirements

```bash
pip install -r requirements.txt
```

### 3. How to Run

```bash
python main.py --mode teacher --epochs 20
👶 Train Student without Distillation
```
```bash
python main.py --mode student --epochs 20
📚 Train Student with Distillation
```

```bash
python main.py --mode distill --temperature 4 --alpha 0.3 --epochs 20
📊 Run Temperature Experiment
```

```bash
python experiment_temperatur.py
✨ Running the app
```

This will:

Run knowledge distillation with multiple temperature values

Save results to temperature_results.json

Generate a graph temperature_analysis.png
